"""Independent single attacks. MP3 requires ffmpeg on PATH."""
import subprocess
import numpy as np
import torch
import torchaudio
from .models import true_st


def transform(x, name, st=None, seed=0):
    if name == 'NA':
        return x
    if name == 'ST':
        return true_st(st, x)
    if name == 'lowpass':
        return torchaudio.functional.lowpass_biquad(x, 16000, 3400)
    if name == 'gaussian':
        rng = torch.Generator(device=x.device).manual_seed(seed)
        noise = torch.randn(x.shape, device=x.device, generator=rng)
        noise = noise / noise.square().mean().sqrt().clamp_min(1e-12)
        return (x + noise * x.square().mean().sqrt() * .1).clamp(-1, 1)
    if name == 'mp3':
        raw = x.flatten().detach().cpu().numpy().astype('<f4').tobytes()
        encoded = subprocess.run(['ffmpeg', '-v', 'error', '-f', 'f32le', '-ar', '16000', '-ac', '1', '-i', 'pipe:0', '-c:a', 'libmp3lame', '-b:a', '32k', '-f', 'mp3', 'pipe:1'], input=raw, capture_output=True, check=True).stdout
        decoded = subprocess.run(['ffmpeg', '-v', 'error', '-f', 'mp3', '-i', 'pipe:0', '-f', 'f32le', '-ar', '16000', '-ac', '1', 'pipe:1'], input=encoded, capture_output=True, check=True).stdout
        return torch.from_numpy(np.frombuffer(decoded, dtype='<f4').copy()).to(x.device).view(1, 1, -1)
    raise ValueError(name)
