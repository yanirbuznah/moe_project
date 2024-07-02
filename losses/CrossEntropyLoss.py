from torch import nn

from . import Loss


class CrossEntropyLoss(Loss):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(*args, **kwargs)


    def __call__(self, *args, **kwargs):
        logits = kwargs['logits']   # cross entropy loss expect logits before softmax
        y_true = next(kwargs[y_true] for y_true in self.possible_y_true if y_true in kwargs.keys())
        self._calc(logits, y_true)
        return self.stat

    def _calc(self, y_pred, y_true):
        self.stat = self.loss(y_pred, y_true)
