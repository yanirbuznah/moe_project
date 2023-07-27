import torch

from metrics.Metric import Metric


class Accuracy(Metric):
    def __init__(self):
        self.correct = 0
        self.total = 0

    def __call__(self, *args, **kwargs):
        y_pred, y_true = args
        self.correct += (y_pred == y_true).sum().item()
        self.total += y_true.size(0)

    def compute(self):
        return self.correct / self.total

    def reset(self):
        self.correct = 0
        self.total = 0
