from sklearn.metrics import f1_score

from metrics.ClassificationMetric import ClassificationMetric


class F1(ClassificationMetric):
    def compute(self):
        return f1_score(self.true, self.pred, average='macro', zero_division=0)
