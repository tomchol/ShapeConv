from .base import _Iter_LRScheduler
from .registry import LR_SCHEDULERS
import math


@LR_SCHEDULERS.register_module
class PolyLR(_Iter_LRScheduler):
    """PolyLR
    """

    def __init__(self, optimizer, niter_per_epoch, max_epochs, power=0.9,
                 last_iter=-1, warm_up=0, end_lr=0.0001,):
        self.max_iters = niter_per_epoch * max_epochs
        self.power = power
        self.warm_up = warm_up
        self.end_lr = end_lr
        super().__init__(optimizer, niter_per_epoch, last_iter)

    def get_lr(self):
        if self.last_iter < self.warm_up:
            multiplier = (self.last_iter / float(self.warm_up)) ** self.power
        else:
            multiplier = (1 - self.last_iter / float(
                self.max_iters)) ** self.power

        lrs = []
        for base_lr in self.base_lrs:
            lr = (base_lr - self.end_lr) * multiplier + self.end_lr
            lrs.append(lr)
        return lrs


@LR_SCHEDULERS.register_module
class CosinusLR(_Iter_LRScheduler):
    """CosinusLR
    """

    def __init__(self, optimizer, niter_per_epoch, max_epochs, power=0.0001,
                 last_iter=-1, warm_up=4, end_lr=25,):
        self.max_iters = niter_per_epoch * max_epochs
        self.power = power
        self.warm_up = warm_up * niter_per_epoch
        self.end_lr = int(end_lr) * niter_per_epoch
        super().__init__(optimizer, niter_per_epoch, last_iter)

    def get_lr(self):
        if self.last_iter < self.warm_up:
            multiplier = (self.last_iter + 1)/self.warm_up
        else:
            multiplier = 1

        lrs = []
        for base_lr in self.base_lrs:
            if self.last_iter < self.end_lr:
                lrs.append(self.power * multiplier)
            else:
                lrs.append(self.power * (math.cos((self.last_iter-self.end_lr)/(self.max_iters - self.end_lr)*math.pi)+1)/2)
        return lrs
