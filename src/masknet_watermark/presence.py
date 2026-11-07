"""Fixed-key presence readout over a frozen watermark decoder's evidence."""
import torch
from torch import nn

class KeyedPresenceReadout(nn.Module):
    """A shared chunk network followed by symmetric pooling and a score correction."""
    def __init__(self, key_bits):
        super().__init__()
        if len(key_bits) != 16 or any(v not in (0, 1) for v in key_bits):
            raise ValueError("Expected sixteen key bits")
        self.register_buffer("key_targets", torch.tensor([
            sum(int(key_bits[i * 4 + j]) << j for j in range(4))
            for i in range(4)
        ], dtype=torch.long))
        self.set_net = nn.Sequential(nn.Linear(3, 8), nn.GELU(), nn.Linear(8, 8), nn.GELU())
        self.head = nn.Sequential(nn.Linear(20, 8), nn.GELU(), nn.Linear(8, 1))

    def forward(self, chunks, frame_summaries):
        probability = chunks.softmax(-1)
        indices = self.key_targets[None, :, None].expand(chunks.shape[0], -1, 1)
        keyed = probability.gather(-1, indices).squeeze(-1)
        other = chunks.clone().scatter(-1, indices, float("-inf"))
        margin = chunks.gather(-1, indices).squeeze(-1) - other.amax(-1)
        odds = torch.logit(keyed.clamp(1e-6, 1 - 1e-6))
        entropy = -(probability * probability.clamp_min(1e-12).log()).sum(-1) / 2.772588722239781
        features = torch.stack([odds.clamp(-8, 8) / 8, margin.clamp(-8, 8) / 8, entropy], -1)
        base = torch.logit(keyed.mean(-1).clamp(1e-6, 1 - 1e-6))
        chunk_features = self.set_net(features)
        pooled = torch.cat([chunk_features.mean(1), chunk_features.amax(1), frame_summaries], -1)
        return base + 4 * torch.tanh(self.head(pooled).squeeze(-1))
