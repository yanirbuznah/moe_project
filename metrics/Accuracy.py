from metrics.ClassificationMetric import ClassificationMetric


class Accuracy(ClassificationMetric):
    def __init__(self):
        super().__init__()

    def compute(self):
        return sum([1 for p, t in zip(self.pred, self.true) if p == t]) / len(self.true)
