from metrics.Metric import Metric
from sklearn.metrics import recall_score


class Recall(Metric):
    def __init__(self):
        self.pred = []
        self.true = []

    def __call__(self, *args, **kwargs):
        y_pred, y_true = args
        self.pred.extend(y_pred)
        self.true.extend(y_true)
        return recall_score(self.true, self.pred, average='macro', zero_division=0)

    def compute(self):
        return recall_score(self.true, self.pred, average='macro', zero_division=0)

    def reset(self):
        self.pred = []
        self.true = []
