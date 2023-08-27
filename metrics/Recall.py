from sklearn.metrics import recall_score

from metrics.ClassificationMetric import ClassificationMetric


class Recall(ClassificationMetric):
    def compute(self):
        return recall_score(self.true, self.pred, average='macro', zero_division=0)
