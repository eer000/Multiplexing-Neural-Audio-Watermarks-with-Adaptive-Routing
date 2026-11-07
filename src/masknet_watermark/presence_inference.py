"""Self-contained fixed-key embedding and presence verification."""
from pathlib import Path
from contextlib import contextmanager
import hashlib
import json
import torch
from .presence import KeyedPresenceReadout
from .encoder_presence import EncoderPresenceReadout

@contextmanager
def _backbone_execution():
    prior = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=False, allow_tf32=True):
            with torch.jit.optimized_execution(False):
                yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prior

class MaskNet:
    FORMAT = "masknet-presence-inference-v3"

    def __init__(self, directory, device="cuda", operating_point=None):
        self.directory = Path(directory).resolve()
        self.device = torch.device(device)
        if self.device.type != "cuda" or self.device.index not in (None, 0):
            raise ValueError("Use a CUDA device exposed as device 0 for this bundle")
        meta = json.loads((self.directory / "model.json").read_text())
        if meta.get("format") not in (self.FORMAT, "masknet-presence-inference-v2"):
            raise ValueError("Unsupported model bundle format")
        self.metadata = meta
        self.evidence_kind = meta.get("evidence_kind", "decoder-evidence")
        if self.evidence_kind not in ("decoder-evidence", "encoder-only"):
            raise ValueError("Unsupported evidence component")
        self.sample_rate = int(meta["sample_rate"])
        self.segment_samples = int(meta["segment_samples"])
        self.score_sample_counts = tuple(meta.get("score_sample_counts", (9600, 24000, 96000)))
        if self.sample_rate != 16000 or self.segment_samples != 96000:
            raise ValueError("Unsupported audio dimensions")
        if self.score_sample_counts != (9600, 24000, 96000):
            raise ValueError("Unsupported detection contexts")
        self.query_ids = tuple(meta["query_ids"])
        if len(self.query_ids) not in (1, 2) or not meta.get("model_id") or not meta.get("calibration_id"):
            raise ValueError("A frozen model, calibration and query are required")
        self.auxiliary_native_mean = bool(meta.get("auxiliary_native_mean", False))
        if self.auxiliary_native_mean != (len(self.query_ids) == 2):
            raise ValueError("Query set differs from the declared inference event")
        self.operating_point = operating_point or meta.get("default_operating_point", "worstcase_0.01")
        points = meta.get("operating_points")
        if points is not None:
            if self.operating_point not in points:
                raise ValueError("Unknown frozen operating point")
            thresholds = points[self.operating_point]["thresholds"]
        else:
            if operating_point is not None:
                raise ValueError("This bundle has one fixed operating point")
            thresholds = meta["thresholds"]
        self.thresholds = torch.tensor(thresholds, dtype=torch.float64)
        if self.thresholds.shape != (len(self.query_ids),) or not torch.isfinite(self.thresholds).all():
            raise ValueError("Invalid calibrated threshold")
        self.score_scales = torch.tensor(meta.get("query_score_scales", [1.] * len(self.query_ids)), dtype=torch.float64)
        if self.score_scales.shape != self.thresholds.shape or not torch.isfinite(self.score_scales).all() or not (self.score_scales > 0).all():
            raise ValueError("Invalid fixed query scales")
        self.residual_scale = float(meta["residual_scale"])
        if not 0 < self.residual_scale <= 1:
            raise ValueError("Invalid embedding scale")
        self.embedder = torch.jit.load(str(self._file(meta["embedder"])), map_location=self.device).eval()
        self.evidence = torch.jit.load(str(self._file(meta["evidence"])), map_location=self.device).eval()
        spec = meta["readout"]
        self.family = spec["family"]
        if self.auxiliary_native_mean:
            if self.evidence_kind != "decoder-evidence" or self.family != "encoder":
                raise ValueError("The two-query bundle requires encoder and native decoder evidence")
            bits = meta["key_bits"]
            if len(bits) != 16 or any(v not in (0, 1) for v in bits):
                raise ValueError("Invalid registered inference key")
            self.native_targets = torch.tensor([[sum(int(bits[k*4+j]) << j for j in range(4)) for k in range(4)]], device=self.device)
        if self.evidence_kind == "encoder-only" and self.family != "encoder":
            raise ValueError("This component requires an encoder presence readout")
        if self.family == "chunk":
            self.readout = KeyedPresenceReadout(meta["key_bits"])
        elif self.family == "encoder":
            self.readout = EncoderPresenceReadout(spec["kind"]).to(self.device)
        else:
            raise ValueError("Unknown presence architecture")
        state = torch.load(self._file(spec), map_location="cpu", weights_only=True)
        if self.family == "chunk" and not torch.equal(state["key_targets"], self.readout.key_targets):
            raise ValueError("Readout key differs from the bundle")
        self.readout.load_state_dict(state, strict=True)
        self.readout.eval()
        for module in (self.embedder, self.evidence, self.readout):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def _file(self, spec):
        name = spec["file"]
        if Path(name).name != name or name in ("", ".", ".."):
            raise ValueError("Weight filenames must be local basenames")
        path = (self.directory / name).resolve()
        if path.parent != self.directory or hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
            raise ValueError("Weight checksum mismatch")
        return path

    def _audio(self, waveform, sample_rate, exact_segment=True):
        if sample_rate != self.sample_rate:
            raise ValueError("Audio sample rate differs from the bundle")
        if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3 or waveform.shape[:2] != (1, 1):
            raise ValueError("Expected one mono audio segment")
        samples = waveform.shape[-1]
        if exact_segment and samples != self.segment_samples:
            raise ValueError("Embedding expects six seconds of audio")
        if not exact_segment and samples not in self.score_sample_counts:
            raise ValueError("Detection expects 0.6, 1.5 or six seconds of audio")
        waveform = waveform.to(device=self.device, dtype=torch.float32)
        if not torch.isfinite(waveform).all():
            raise ValueError("Audio contains nonfinite samples")
        return waveform

    @torch.no_grad()
    def embed(self, waveform, sample_rate):
        x = self._audio(waveform, sample_rate)
        with _backbone_execution():
            native = self.embedder(x)
        marked = x + self.residual_scale * (native - x)
        if marked.shape != x.shape or not torch.isfinite(marked).all():
            raise RuntimeError("Embedding returned invalid audio")
        return marked

    @torch.no_grad()
    def score(self, waveform, sample_rate):
        x = self._audio(waveform, sample_rate, exact_segment=False)
        with _backbone_execution():
            evidence = self.evidence(x)
        if self.evidence_kind == "encoder-only":
            embedding, frames, chunks = evidence, None, None
        else:
            embedding, frames, chunks = evidence
        if self.family == "chunk":
            probability = frames.sigmoid().flatten()
            summaries = torch.stack([probability.mean(), probability.std(unbiased=False), probability.min(), probability.max()])
            score = self.readout(chunks.cpu(), summaries.cpu()[None])
        else:
            prior = torch.backends.cuda.matmul.allow_tf32
            try:
                torch.backends.cuda.matmul.allow_tf32 = False
                with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True, allow_tf32=False):
                    score = self.readout(embedding)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = prior
        if score.shape != (1,) or not torch.isfinite(score).all():
            raise RuntimeError("Detector returned invalid scores")
        if self.auxiliary_native_mean:
            native = chunks.softmax(-1).gather(-1, self.native_targets.unsqueeze(-1)).mean()
            score = torch.cat([score, native.reshape(1)])
        if score.shape != (len(self.query_ids),) or not torch.isfinite(score).all():
            raise RuntimeError("Detector returned invalid registered scores")
        values = score.detach().cpu().to(torch.float64)
        decisions = values >= self.thresholds
        return {"score": dict(zip(self.query_ids, values.tolist())),
                "detected": bool(decisions.any()),
                "detection_margin": float(((values - self.thresholds) / self.score_scales).max()),
                "query_decisions": dict(zip(self.query_ids, decisions.tolist()))}

    def embed_audio(self, waveform, sample_rate):
        """Embed arbitrary-duration mono audio and return a length-matched CPU tensor."""
        from .long_audio import embed_audio
        return embed_audio(self, waveform, sample_rate)

    def score_audio(self, waveform, sample_rate):
        """Return timestamped, fixed-policy decisions for each analysis window."""
        from .long_audio import score_audio
        return score_audio(self, waveform, sample_rate)
