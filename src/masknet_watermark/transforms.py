"""Single-channel perturbations used only by this model's paired evaluation."""
from __future__ import annotations
import gc,hashlib,subprocess,tempfile
from pathlib import Path
import numpy as np
from scipy.signal import butter,sosfiltfilt,fftconvolve
import torch
import torchaudio.functional as AF

def digest(path):
 return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def state_hash(module):
 h=hashlib.sha256()
 for name,value in sorted(module.state_dict().items()):
  h.update(name.encode());h.update(str((tuple(value.shape),value.dtype)).encode());h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
 return h.hexdigest()
def freeze(module):
 module.eval()
 for p in module.parameters():p.requires_grad_(False)
 return module
def array(x):
 if isinstance(x,torch.Tensor):x=x.detach().cpu().numpy()
 x=np.asarray(x,dtype=np.float32).reshape(-1)
 if not len(x) or not np.isfinite(x).all():raise ValueError('Invalid audio')
 return x
class Channels:
    def __init__(self, assets, device='cuda'):
        self.assets=assets;self.device=torch.device(device);self.codec=self.codec_kind=None;self.identities={};self.last_details={}
        self.ffmpeg_version=subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0]
    def tensor(self,x):
        return torch.as_tensor(x,dtype=torch.float32,device=self.device).reshape(1,1,-1)
    def unload(self):
        if self.codec is not None and state_hash(self.codec)!=self.identities[self.codec_kind]['state_sha256']:raise ValueError('Frozen codec changed')
        self.codec=self.codec_kind=None;gc.collect();torch.cuda.empty_cache()
    def load_codec(self,family):
        if self.codec_kind==family:return
        self.unload();spec=self.assets[family]
        if family=='ST':
            from speechtokenizer import SpeechTokenizer
            if digest(spec['config'])!=spec['config_SHA256'] or digest(spec['checkpoint'])!=spec['checkpoint_SHA256']:raise ValueError('Codec asset checksum mismatch')
            self.codec=SpeechTokenizer.load_from_checkpoint(spec['config'],spec['checkpoint']).to(self.device)
        elif family=='EN':
            from transformers import EncodecModel
            self.codec=EncodecModel.from_pretrained(spec['directory'],local_files_only=True).to(self.device)
        elif family=='DA':
            import dac
            if digest(spec['checkpoint'])!=spec['checkpoint_SHA256']:raise ValueError('Codec asset checksum mismatch')
            self.codec=dac.DAC.load(spec['checkpoint']).to(self.device)
        else:raise ValueError('Unknown codec')
        freeze(self.codec);actual=state_hash(self.codec)
        if actual!=spec['state_SHA256']:raise ValueError('Codec state mismatch')
        self.codec_kind=family;self.identities[family]={'state_sha256':actual}
    def apply(self,x,condition,seed,clean_reference):
        if condition['family']!='DA':return self._apply(x,condition,seed,clean_reference)
        prior=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]),torch.backends.cudnn.flags(enabled=True,benchmark=False,deterministic=True,allow_tf32=False):
            try:
                torch.use_deterministic_algorithms(True)
                if self.codec_kind!='DA':
                    for _ in range(5):
                        torch.manual_seed(seed%(2**32));torch.cuda.manual_seed_all(seed%(2**32));self._apply(x,condition,seed,clean_reference)
                torch.manual_seed(seed%(2**32));torch.cuda.manual_seed_all(seed%(2**32));return self._apply(x,condition,seed,clean_reference)
            finally:torch.use_deterministic_algorithms(prior,warn_only=warn)
    def ffmpeg(self, x, codec, bitrate):
        # Seekable files retain encoder-delay/gapless metadata. All temporary
        # audio stays on the server and is removed after this single transform.
        with tempfile.TemporaryDirectory(prefix="taslp_codec_") as tmp:
            path = str(Path(tmp)/("audio.mp3" if codec == "libmp3lame" else "audio.ogg"))
            subprocess.run(["ffmpeg", "-v", "error", "-threads", "1", "-f", "f32le", "-ar", "16000",
                            "-ac", "1", "-i", "pipe:0", "-c:a", codec,
                            "-b:a", f"{bitrate}k", path], input=np.asarray(x, np.float32).tobytes(),
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45, check=True)
            decoded = subprocess.run(["ffmpeg", "-v", "error", "-threads", "1", "-i", path,
                                      "-f", "f32le", "-ar", "16000", "-ac", "1", "pipe:1"],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45, check=True).stdout
        y = np.frombuffer(decoded, np.float32).copy()
        if abs(len(y)-len(x)) > 32:
            raise RuntimeError(f"Unexpected codec length: {len(y)} vs {len(x)}")
        return y[:len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x)-len(y)))

    def _apply(self, x, condition, seed, clean_reference):
        self.last_details = {}
        family, p = condition["family"], condition["parameters"]
        rng = np.random.default_rng(seed)
        x = np.asarray(x, np.float32)
        if family == "NA":
            y = x.copy()
        elif family in {"GN", "UN"}:
            sigma = np.sqrt(np.mean(np.asarray(clean_reference, np.float64)**2) / 10**(p["snr_db"]/10))
            n = rng.normal(0, 1, len(x)) if family == "GN" else rng.uniform(-np.sqrt(3), np.sqrt(3), len(x))
            y = x + sigma*n
        elif family == "ZM":
            y = x.copy()
            y[rng.choice(len(x), int(len(x)*p["erase_fraction"]), replace=False)] = 0
        elif family == "FT":
            spectrum = np.fft.fft(x)
            spectrum[rng.choice(len(x), int(len(x)*p["erase_fraction"]), replace=False)] = 0
            y = np.fft.ifft(spectrum).real
        elif family == "EC":
            delay = int(16*p["delay_ms"])
            y = x.copy()
            y[delay:] += p["gain"]*x[:-delay]
        elif family == "LP":
            y = sosfiltfilt(butter(p["order"], p["cutoff_hz"], fs=16000, btype="low", output="sos"), x)
        elif family == "CROP":
            n = int(len(x)*p["retain_fraction"])
            start = int(rng.integers(0, len(x)-n+1))
            y = x[start:start+n]
        elif family in {"MP", "OP"}:
            y = self.ffmpeg(x, "libmp3lame" if family == "MP" else "libopus", p["bitrate_kbps"])
        elif family == "RV":
            n = p["length_samples"]
            h = rng.normal(size=n)*np.exp(-np.arange(n)/(n*p["decay_factor"]))
            for i in range(1, int(n*.1)):
                pos = int(i*.1*n)
                if pos < n:
                    h[pos] += .5*np.exp(-i/int(n*.1))
            h /= np.max(np.abs(h))
            # Freeze normalization on clean channel, so clean/marked get the same gain.
            scale = max(float(np.max(np.abs(fftconvolve(clean_reference, h, mode="same")))), 1e-12)
            y = fftconvolve(x, h, mode="same")/scale
        elif family in {"EN", "ST", "DA", "ST_RVQ"}:
            self.load_codec('ST' if family=='ST_RVQ' else family)
            with torch.no_grad():
                t = self.tensor(x)
                if family in {"ST","ST_RVQ"}:
                    if self.codec.sample_rate != 16000:raise ValueError('Fixed SpeechTokenizer attack requires native16kHz')
                    codes = self.codec.encode(t)
                    if codes.ndim!=3 or codes.shape[0]!=8:raise ValueError('Fixed SpeechTokenizer attack requires all8RVQ streams')
                    self.last_details={'codec':'SpeechTokenizer','native_sample_rate':16000,'input_sample_rate':16000,'codes_shape':list(codes.shape),'rvq_streams':int(codes.shape[0]),'all_RVQ':True}
                    if family=='ST_RVQ':
                        n=p['rvq_prefix_streams']
                        if type(n) is not int or n not in [1,2,4] or p['available_streams']!=8:raise ValueError('Unregistered RVQ prefix')
                        codes=codes[:n]
                        self.last_details.update(rvq_streams=n,all_RVQ=False,selection='first_n_residual_quantizers',applied_codes_shape=list(codes.shape))
                        self.identities[condition['id']]={**self.identities['ST'],'rvq_layers':'prefix_'+str(n),'available_streams':8,'applied_streams':n}
                    y = array(self.codec.decode(codes))
                elif family == "EN":
                    t = AF.resample(t, 16000, 24000)
                    encoded = self.codec.encode(t, bandwidth=p["bandwidth_kbps"])
                    decoded = self.codec.decode(encoded.audio_codes, encoded.audio_scales)[0]
                    y = array(AF.resample(decoded, 24000, 16000))
                else:
                    native_rate = self.codec.sample_rate
                    t = AF.resample(t, 16000, native_rate)
                    t = self.codec.preprocess(t, native_rate)
                    y = array(AF.resample(self.codec(t)["audio"], native_rate, 16000))
                y = y[:len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x)-len(y)))
        else:
            raise ValueError(family)
        return array(y)
