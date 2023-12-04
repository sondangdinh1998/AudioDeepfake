from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class NoamScheduler(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        model_size: int,
        warmup_steps: int,
    ):
        self.warmup_steps = warmup_steps
        self.normalize = model_size ** (-0.5)
        super(NoamScheduler, self).__init__(optimizer, last_epoch=-1)

    def get_lr(self):
        step = max(1, self.last_epoch)
        new_lrs = [self._noam_annealing(lr, step) for lr in self.base_lrs]
        return new_lrs

    def _noam_annealing(self, base_lr, step):
        return (
            base_lr
            * self.normalize
            * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
        )
