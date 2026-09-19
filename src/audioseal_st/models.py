import torch
from torch import nn


def local_noise_limit(clean, residual, minimum_snr=26):
    """Smooth pointwise residual limit tied to a 40ms clean RMS envelope.

    No absolute floor that injects noise in silence. This is not a validated
    auditory masking model or a guarantee of per-window SNR.
    """
    power=torch.nn.functional.avg_pool1d(clean.square(),641,stride=1,padding=320)
    limit=2*10**(-minimum_snr/20)*power.clamp_min(0).sqrt()
    return limit*torch.tanh(residual/limit.clamp_min(1e-8))


class ResidualAdapter(nn.Module):
    """Frozen AudioSeal plus zero-initialized waveform residual correction."""
    def __init__(self, base):
        super().__init__()
        self.base=base
        for p in base.parameters():p.requires_grad=False
        self.correction=nn.Sequential(nn.Conv1d(2,32,15,padding=7),nn.SiLU(),nn.Conv1d(32,32,15,padding=7),nn.SiLU(),nn.Conv1d(32,1,15,padding=7))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def get_watermark(self,x,sample_rate,message):
        self.base.eval()
        with torch.no_grad():d=self.base.get_watermark(x,sample_rate,message=message)
        return d+self.correction(torch.cat([x,d],dim=1))


class MaskNet(nn.Module):
    """Explicit architecture; checkpoint metadata must specify output activation.

    relu: nonnegative, unbounded. sigmoid: [0,1]. No implicit conversion between
    these architectures is permitted, even if state dictionary shapes match.
    """
    def __init__(self, channels=128, layers=5, kernel_size=15, activation="relu"):
        super().__init__()
        if activation not in {"relu", "sigmoid"} or layers < 2 or kernel_size % 2 != 1:
            raise ValueError("Invalid MaskNet architecture")
        blocks = []
        for i in range(layers):
            blocks.append(nn.Conv1d(1 if i == 0 else channels,
                                    2 if i == layers-1 else channels,
                                    kernel_size, padding=kernel_size//2))
            if i < layers-1:
                blocks.append(nn.ReLU())
        self.conv_layers = nn.Sequential(*blocks)
        self.activation = activation

    def forward(self, x):
        z = self.conv_layers(x)
        return torch.relu(z) if self.activation == "relu" else torch.sigmoid(z)


def audioseal_outputs(detector, x):
    output = detector(x, sample_rate=16000)
    # Official AudioSeal returns two outputs; historical local fork adds a third.
    return output[0][:, 1, :], output[1]


def true_st(model, x):
    if model.sample_rate != 16000:
        raise ValueError("This protocol requires the 16 kHz SpeechTokenizer checkpoint")
    model.eval()
    with torch.no_grad():
        y = model.decode(model.encode(x))
    if y.shape[-1] < x.shape[-1]:
        raise ValueError("Tokenizer unexpectedly shortened the waveform")
    return y[..., :x.shape[-1]]


def shape_residual(clean, residual, minimum_snr=22.0, spectral_ratio=0.2):
    """Differentiable carrier-dependent limits on quiet and spectral regions.

    This is an experimental quality guard, not a validated auditory masking
    model. It must be applied identically during training and inference.
    """
    import torch.nn.functional as F
    envelope = F.avg_pool1d(clean.abs(), 513, stride=1, padding=256)
    gate = (envelope/.01).clamp(0, 1)
    residual = residual*gate
    window = torch.hann_window(512, device=clean.device, dtype=clean.dtype)
    x = torch.stft(clean.squeeze(1), 512, 128, window=window, return_complex=True)
    r = torch.stft(residual.squeeze(1), 512, 128, window=window, return_complex=True)
    reference = F.avg_pool2d(x.abs().unsqueeze(1), (7, 5), stride=1, padding=(3, 2)).squeeze(1)
    limit = spectral_ratio*reference
    r = r*(limit/r.abs().clamp_min(1e-8)).clamp(max=1)
    residual = torch.istft(r, 512, 128, window=window, length=clean.shape[-1]).unsqueeze(1)*gate
    signal = clean.square().mean(-1, keepdim=True)
    power = residual.square().mean(-1, keepdim=True).clamp_min(1e-12)
    scale = (signal*10**(-minimum_snr/10)/power).clamp_min(1e-12).sqrt().clamp(max=1)
    return residual*scale


class GatedLongContextProxy(nn.Module):
    """Long-context residual surrogate for codec/tokenizer reconstruction."""

    def __init__(self, channels: int = 96, layers: int = 12, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.in_proj = nn.Conv1d(1, channels, kernel_size, padding=pad)
        self.blocks = nn.ModuleList()
        for i in range(layers):
            dilation = 2 ** (i % 10)
            block_pad = dilation * pad
            self.blocks.append(nn.ModuleDict({
                "filter": nn.Conv1d(channels, channels, kernel_size,
                                    padding=block_pad, dilation=dilation),
                "gate": nn.Conv1d(channels, channels, kernel_size,
                                  padding=block_pad, dilation=dilation),
                "mix": nn.Conv1d(channels, channels, 1),
            }))
        self.out_proj = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(channels, channels // 2, 1),
            nn.SiLU(),
            nn.Conv1d(channels // 2, 1, kernel_size, padding=pad),
        )

    def forward(self, x):
        h = self.in_proj(x)
        for block in self.blocks:
            update = torch.tanh(block["filter"](h)) * torch.sigmoid(block["gate"](h))
            h = (h + block["mix"](update)) * (2.0 ** -0.5)
        return x + self.out_proj(h)
