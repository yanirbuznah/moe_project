import numpy as np
import torch

from metrics.Metric import Metric
from utils.singleton_meta import SingletonMeta


class ClassificationMetricPreprocessor(Metric, metaclass=SingletonMeta):
    def __init__(self):
        super().__init__()
        self.reset()

    def __call__(self, *args, **kwargs):
        y_pred = next(kwargs[y_pred] for y_pred in self.possible_y_pred if y_pred in kwargs.keys())
        y_true = next(kwargs[y_true] for y_true in self.possible_y_true if y_true in kwargs.keys())
        y_pred = y_pred.detach().cpu().numpy() if isinstance(y_pred, torch.Tensor) else y_pred
        y_true = y_true.detach().cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
        self.pred = np.append(self.pred, y_pred.argmax(axis=1))
        self.true = np.append(self.true, y_true)

    def reset(self):
        self.pred = np.array([])
        self.true = np.array([])

class ClassificationMetric(Metric):
    def __init__(self):
        super().__init__()
        self.classification_preprocessor = ClassificationMetricPreprocessor()
        self.reset()

    def reset(self):
        self.classification_preprocessor.reset()

    def __call__(self, *args, **kwargs):
        self.classification_preprocessor(*args, **kwargs)

    @property
    def pred(self):
        return self.classification_preprocessor.pred

    @property
    def true(self):
        return self.classification_preprocessor.true

