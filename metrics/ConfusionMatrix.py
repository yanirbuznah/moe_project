import pandas as pd
import torch

from metrics.ClassificationMetric import ClassificationMetric
from sklearn.metrics import confusion_matrix


class ConfusionMatrix(ClassificationMetric):
    def compute(self):
        return pd.DataFrame(confusion_matrix(self.true, self.pred))


    @staticmethod
    def compute_from_y_pred_y_true(y_true,y_pred):
        return pd.DataFrame(confusion_matrix(y_true, y_pred))