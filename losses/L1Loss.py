from torch import nn

from . import Loss


class L1Loss(Loss):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.loss = nn.L1Loss(*args, **kwargs)
        self.possible_y_pred = ['y_pred', 'output', 'y_hat', 'out']
        self.possible_y_true = ['target', 'y_true']

    def __call__(self, *args, **kwargs):
        y_pred = next(kwargs[y_pred] for y_pred in self.possible_y_pred if y_pred in kwargs.keys())
        y_true = next(kwargs[y_true] for y_true in self.possible_y_true if y_true in kwargs.keys())
        self.stat = self.loss(y_pred, y_true)
        return self.stat

    def _calc(self, y_pred, y_true):
        self.stat = self.loss(y_pred, y_true)
