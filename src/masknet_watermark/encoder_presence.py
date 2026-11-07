"""Presence classification from a frozen watermark encoder sequence."""
import torch
from torch import nn

class EncoderPresenceReadout(nn.Module):
    def __init__(self, kind):
        super().__init__()
        if kind not in ("linear", "temporal_mask"):
            raise ValueError("Unknown encoder readout")
        self.kind = kind
        self.norm = nn.LayerNorm(1024, elementwise_affine=False)
        if kind == "linear":
            self.classifier = nn.Linear(2048, 1)
        else:
            self.mask = nn.Sequential(nn.Conv1d(1024, 16, 1), nn.GELU(), nn.Conv1d(16, 1, 1))
            self.classifier = nn.Sequential(nn.Linear(3072, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, embedding):
        feature = self.norm(embedding.transpose(1, 2)).transpose(1, 2)
        mean = feature.mean(-1)
        std = feature.std(-1, unbiased=False)
        if self.kind == "linear":
            pooled = torch.cat([mean, std], 1)
        else:
            mask = self.mask(feature).softmax(-1)
            pooled = torch.cat([mean, std, (mask * feature).sum(-1)], 1)
        return self.classifier(pooled).squeeze(-1)
