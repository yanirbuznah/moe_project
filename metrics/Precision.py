from metrics.Metric import Metric
from sklearn.metrics import precision_score

class Precision(Metric):
    def __init__(self):
        self.pred = []
        self.true = []

    def __call__(self, *args, **kwargs):
        y_pred, y_true = self._preprocess_args(*args)
        self.pred.extend(y_pred)
        self.true.extend(y_true)
        return precision_score(self.true, self.pred, average='macro', zero_division=0)

    def compute(self):
        return precision_score(self.true, self.pred, average='macro', zero_division=0)

    def reset(self):
        self.pred = []
        self.true = []