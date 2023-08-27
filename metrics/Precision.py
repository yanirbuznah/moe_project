from sklearn.metrics import precision_score

from metrics.ClassificationMetric import ClassificationMetric


class Precision(ClassificationMetric):
    def compute(self):

        return precision_score(self.true, self.pred, average='macro', zero_division=0)
