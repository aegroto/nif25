import numpy
import math
import torch


class CustomScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        T_0,
        warmup_iterations=0,
        peak_decay_factor=1.0,
        floor_decay_factor=1.0,
        T_mult=1,
        floor_lr=0,
        warmup_slope=1.0,
        decay_slope=1.0,
        last_epoch=-1,
        verbose=False,
    ):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.floor_lr = floor_lr
        self.T_cur = last_epoch
        self.warmup_iterations = warmup_iterations
        self.peak_decay = 1.0
        self.peak_decay_factor = peak_decay_factor

        self.floor_decay = 1.0
        self.floor_decay_factor = floor_decay_factor

        warmup_progression = numpy.linspace(0.0, 1.0, num=warmup_iterations) ** warmup_slope
        decay_progression = numpy.linspace(1.0, 0.0, num=(T_0 - warmup_iterations) + 1) ** decay_slope

        self.progression_factors = warmup_progression.tolist()[:-1] + decay_progression.tolist()

        super(CustomScheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            print(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        current_floor_lr = self.floor_lr * self.floor_decay
        current_factor = self.progression_factors[self.T_cur]

        result = [
            current_floor_lr + (peak_lr * self.peak_decay - current_floor_lr) * (current_factor)
            for peak_lr in self.base_lrs
        ]

        return result

    def step(self, epoch=None):
        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
                self.peak_decay *= self.peak_decay_factor
                self.floor_decay *= self.floor_decay_factor
        else:
            if epoch < 0:
                raise ValueError(
                    "Expected non-negative epoch, but got {}".format(epoch)
                )
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(
                        math.log(
                            (epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult
                        )
                    )
                    self.T_cur = epoch - self.T_0 * (self.T_mult**n - 1) / (
                        self.T_mult - 1
                    )
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        self.last_epoch = math.floor(epoch)

        class _enable_get_lr_call:
            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
                param_group, lr = data
                param_group["lr"] = lr
                self.print_lr(self.verbose, i, lr, epoch)

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
