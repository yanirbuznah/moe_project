from metrics.Metric import Metric
from sklearn.metrics import f1_score
class F1(Metric):
    def __init__(self):
        self.pred = []
        self.true = []

    def __call__(self, *args, **kwargs):
        y_pred, y_true = args
        self.pred.extend(y_pred)
        self.true.extend(y_true)
        return f1_score(self.true, self.pred, average='macro', zero_division=0) # TODO: average by configuration

    def compute(self):
        return f1_score(self.true, self.pred, average='macro', zero_division=0)

    def reset(self):
        self.pred = []
        self.true = []