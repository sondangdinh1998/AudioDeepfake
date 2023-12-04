import torch
import torch.nn as nn
import torch.nn.functional as F


class OCSoftmax(nn.Module):
    def __init__(
        self,
        alpha: float = 20.0,
        margin_real: float = 0.9,
        margin_fake: float = 0.2,
    ):
        super().__init__()
        self.alpha = alpha
        self.margin_real = margin_real
        self.margin_fake = margin_fake

    def forward(self, scores: torch.Tensor, labels: torch.Tensor):
        scores[labels == 0] = self.margin_real - scores[labels == 0]
        scores[labels == 1] = scores[labels == 1] - self.margin_fake
        loss = F.softplus(self.alpha * scores, beta=1).mean()
        return loss
