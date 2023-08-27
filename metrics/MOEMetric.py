import numpy as np
import torch
from scipy.stats import chi2_contingency
from sklearn.metrics import confusion_matrix

from metrics.Metric import Metric


class MOEMetric(Metric):
    def __init__(self, ):
        super().__init__()
        self.reset()

    def reset(self):
        self.gates = torch.tensor([], dtype=torch.long)
        self.model_index = torch.tensor([], dtype=torch.long)
        self.model_output_class = torch.tensor([], dtype=torch.long)
        self.correct = torch.tensor([], dtype=torch.long)
        self.random_correct = torch.tensor([], dtype=torch.long)
        self.true = torch.tensor([], dtype=torch.long)
        self.num_experts = 0

    def __call__(self, *args, **kwargs):
        y_pred, y_true, routes, counts, model, x = self._parse_args(*args, **kwargs)
        random_routes = torch.randint(0, model.num_experts, (x.shape[0],), device=model.device)
        random_outputs = model.get_experts_output_from_indexes_list(model.get_indexes_list(random_routes), x).argmax(
            dim=-1)
        self.gates = torch.cat((self.gates, kwargs['router_probs'].argmax(dim=-1)), dim=0)
        self.model_index = torch.cat((self.model_index, routes), dim=0)
        self.correct = torch.cat((self.correct, (y_pred == y_true).long()), dim=0)
        self.random_correct = torch.cat((self.random_correct, (random_outputs == y_true).long()), dim=0)
        self.model_output_class = torch.cat((self.model_output_class, y_pred), dim=0)
        self.true = torch.cat((self.true, y_true), dim=0)
        self.num_experts = model.num_experts

    def _parse_args(self, *args, **kwargs):
        y_pred = next(kwargs[y_pred] for y_pred in self.possible_y_pred if y_pred in kwargs.keys()).argmax(dim=-1)
        y_true = next(kwargs[y_true] for y_true in self.possible_y_true if y_true in kwargs.keys())
        routes = next(kwargs[routes_probs] for routes_probs in self.possible_routes_probs if
                      routes_probs in kwargs.keys()).argmax(dim=-1)
        counts = next(kwargs[counts] for counts in self.possible_counts if counts in kwargs.keys())
        model = next(kwargs[model] for model in self.possible_model if model in kwargs.keys())
        x = next(kwargs[x] for x in self.possible_x if x in kwargs.keys())
        return y_pred, y_true, routes, counts, model, x


class RouterVSRandomAcc(MOEMetric):
    def compute(self):
        diff = self.correct.mean(dtype=float) - self.random_correct.mean(dtype=float)
        diff_mean, diff_std = diff.mean().item(), diff.std().item()
        return torch.Tensor([diff_mean, diff_std])


class ConfusionMatrix(MOEMetric):
    def compute(self):
        return confusion_matrix(self.true, self.gates)[:, :self.num_experts]


class PValue(MOEMetric):
    def compute(self):
        conmat = confusion_matrix(self.true, self.gates)[:, :self.num_experts]
        try:
            return chi2_contingency(conmat)[1]
        except Exception as e:
            print('p-value calculation failed with error: ', e)
            print('adding small noise to the confusion matrix')
            return chi2_contingency(conmat + np.random.uniform(1e-10, 1e-5, conmat.shape))[1]
