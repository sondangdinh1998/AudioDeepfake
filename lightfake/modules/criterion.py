import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_det_curve(target_scores, nontarget_scores):
    num_targets = len(target_scores)
    num_nontargets = len(nontarget_scores)

    n_scores = num_targets + num_nontargets
    all_scores = torch.cat((target_scores, nontarget_scores))
    labels = torch.cat((torch.ones(num_targets), torch.zeros(num_nontargets)))

    # Sort labels based on scores
    indices = torch.argsort(all_scores)
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = torch.cumsum(labels, dim=0)
    nontarget_trial_sums = num_nontargets - (
        torch.arange(1, n_scores + 1) - tar_trial_sums
    )

    frr = torch.cat(
        (torch.tensor([0]), tar_trial_sums / num_targets)
    )  # false rejection rates
    far = torch.cat(
        (torch.tensor([1]), nontarget_trial_sums / num_nontargets)
    )  # false acceptance rates
    thresholds = torch.cat(
        (torch.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices])
    )  # Thresholds are the sorted scores

    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = torch.abs(frr - far)
    min_index = torch.argmin(abs_diffs)
    eer = (frr[min_index] + far[min_index]) / 2
    return eer, thresholds[min_index]


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
