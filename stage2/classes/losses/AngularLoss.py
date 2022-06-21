import math

import torch
from torch import Tensor
from torch.nn.functional import normalize

from classes.core.Loss import Loss


class AngularLoss(Loss):
    def __init__(self, device: torch.device):
        super().__init__(device)

    def _compute(self, pred: Tensor, label: Tensor, safe_v: float = 0.999999) -> Tensor:
        dot = torch.clamp(torch.sum(normalize(pred, dim=1) * normalize(label, dim=1), dim=1), -safe_v, safe_v)
        angle = torch.acos(dot) * (180 / math.pi)
        return torch.mean(angle).to(self._device)
