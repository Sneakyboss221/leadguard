"""Signal Quality Assessor (SPEC section 3A): one reliability score q in [0, 1] per lead."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class LeadQualityNet(nn.Module):
    """Shared small 1D CNN applied to every lead independently.

    4 Conv1D blocks (kernel 7, stride 2, BatchNorm, GELU) -> global avg + max pool -> linear -> sigmoid.
    forward(x: (B, 12, T)) -> q: (B, 12)
    """

    def __init__(self, channels: tuple[int, ...] = (16, 32, 64, 64)):
        super().__init__()
        layers, c_in = [], 1
        for c in channels:
            layers += [nn.Conv1d(c_in, c, kernel_size=7, stride=2, padding=3, bias=False),
                       nn.BatchNorm1d(c), nn.GELU()]
            c_in = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(2 * c_in, 1)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 12, T) -> logits (B, 12)."""
        b, c, t = x.shape
        h = self.features(x.reshape(b * c, 1, t))
        h = torch.cat([h.mean(-1), h.amax(-1)], dim=1)
        return self.head(h).view(b, c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(x))


@torch.no_grad()
def estimate_q(model: LeadQualityNet, x: torch.Tensor) -> torch.Tensor:
    """Frozen SQA inference. x: (B, 12, T) -> q: (B, 12)."""
    return model(x)


def freeze(model: nn.Module) -> nn.Module:
    """Eval mode and no gradients."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_sqa(path: str | Path, device: str | torch.device = "cpu") -> LeadQualityNet:
    """Load checkpoints/sqa.pt as a frozen model."""
    ckpt = torch.load(Path(path), map_location=device, weights_only=False)
    model = LeadQualityNet(**ckpt.get("config", {})).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return freeze(model)
