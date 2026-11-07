"""Windowed file handling around the unchanged calibrated inference core."""
import math
import torch
from torch.nn import functional as F

def _host_audio(model, waveform, sample_rate):
    if sample_rate != model.sample_rate:
        raise ValueError("Resample the input to the model sample rate first")
    if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3 or waveform.shape[:2] != (1, 1):
        raise ValueError("Expected one mono audio tensor shaped (1, 1, samples)")
    if waveform.shape[-1] == 0:
        raise ValueError("Audio must contain at least one sample")
    x = waveform.detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(x).all():
        raise ValueError("Audio contains nonfinite samples")
    return x

@torch.no_grad()
def embed_audio(model, waveform, sample_rate):
    """Embed any nonempty duration; six-second core and bounded GPU memory.

    Five-second hops give adjacent six-second windows a one-second residual
    crossfade. Only the final input window is zero-padded, then trimmed.
    Returns a CPU tensor with the exact original sample count.
    """
    x = _host_audio(model, waveform, sample_rate)
    length, window = x.shape[-1], model.segment_samples
    if length <= window:
        padded = F.pad(x, (0, window-length))
        return model.embed(padded, sample_rate).detach().cpu()[..., :length]
    overlap = model.sample_rate
    hop = window-overlap
    starts = [i*hop for i in range(math.ceil((length-window)/hop)+1)]
    residual = torch.zeros_like(x)
    normalizer = torch.zeros_like(x)
    ramp = torch.linspace(0., 1., overlap, dtype=x.dtype).reshape(1, 1, -1)
    for index, start in enumerate(starts):
        stop = min(start+window, length)
        valid = stop-start
        chunk = F.pad(x[..., start:stop], (0, window-valid))
        marked = model.embed(chunk, sample_rate).detach().cpu()
        weight = torch.ones_like(chunk)
        if index > 0:
            weight[..., :overlap] = ramp
        if index < len(starts)-1:
            weight[..., -overlap:] = 1-ramp
        residual[..., start:stop] += (marked-chunk)[..., :valid]*weight[..., :valid]
        normalizer[..., start:stop] += weight[..., :valid]
    if not (normalizer > 0).all():
        raise RuntimeError("Window layout left uncovered samples")
    output = x+residual/normalizer
    if output.shape != x.shape or not torch.isfinite(output).all():
        raise RuntimeError("Windowed embedding returned invalid audio")
    return output

@torch.no_grad()
def score_audio(model, waveform, sample_rate):
    """Return timestamped per-window decisions; no uncalibrated whole-file OR."""
    x = _host_audio(model, waveform, sample_rate)
    length, window = x.shape[-1], model.segment_samples
    segments = []
    for start in range(0, length, window):
        stop = min(start+window, length)
        valid = stop-start
        context = next(v for v in model.score_sample_counts if v >= valid)
        chunk = F.pad(x[..., start:stop], (0, context-valid))
        result = model.score(chunk, sample_rate)
        segments.append({
            "start_sample": start, "end_sample": stop,
            "start_seconds": start/sample_rate, "end_seconds": stop/sample_rate,
            "valid_samples": valid, "scored_samples": context,
            "padding_samples": context-valid,
            "window_kind": "native_context" if valid == context else "zero_padded_context",
            **result,
        })
    return {"model_id": model.metadata["model_id"], "sample_rate": sample_rate,
            "samples": length, "duration_seconds": length/sample_rate,
            "decision_scope": "per_window", "segments": segments}
