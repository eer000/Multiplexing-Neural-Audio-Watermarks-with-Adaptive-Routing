"""Frozen embedding and decoder-evidence components used by this watermark."""
import torch
from torch import nn
from torch.nn import functional as F

@torch.jit.script
def dynamic_conv_padding(x: torch.Tensor, kernel: int, stride: int,
                         padding_total: int, causal: bool, mode: str) -> torch.Tensor:
    length = x.size(-1)
    extra = (stride - ((length - kernel + padding_total) % stride)) % stride
    if causal:
        left, right = padding_total, extra
    else:
        right = padding_total // 2
        left = padding_total - right
        right = right + extra
    tail = 0
    if mode == "reflect":
        maximum = max(left, right)
        if length <= maximum:
            tail = maximum - length + 1
            x = F.pad(x, [0, tail])
    padded = F.pad(x, [left, right], mode, 0.0)
    return padded[..., :padded.size(-1) - tail]

class DynamicSConv1d(nn.Module):
    """The frozen convolution with input-length-dependent scripted padding."""
    def __init__(self, original):
        super().__init__()
        self.conv = original.conv
        layer = original.conv.conv
        self.kernel = int(layer.kernel_size[0])
        self.stride = int(layer.stride[0])
        self.padding_total = (self.kernel - 1) * int(layer.dilation[0]) - (self.stride - 1)
        self.causal = bool(original.causal)
        self.pad_mode = str(original.pad_mode)

    def forward(self, waveform):
        return self.conv(dynamic_conv_padding(waveform, self.kernel, self.stride,
                                             self.padding_total, self.causal, self.pad_mode))

class FrozenEmbedding(nn.Module):
    def __init__(self, codec, watermark_generator, fusion, key_bits):
        super().__init__()
        self.codec = codec
        self.watermark_generator = watermark_generator
        self.fusion = fusion
        self.register_buffer("message", torch.tensor([key_bits], dtype=torch.long))

    def forward(self, waveform):
        feature = self.codec.encode(waveform)
        marked_feature = self.watermark_generator(feature, self.message)
        marked_audio = self.codec.decode(marked_feature)
        return self.fusion(waveform, marked_audio)

class FrozenEvidence(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, waveform):
        embedding = self.encoder(waveform)
        frames, chunks, _ = self.decoder.detect_watermark(embedding)
        return embedding, frames, chunks
