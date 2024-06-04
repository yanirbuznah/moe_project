import numpy as np
import pandas as pd
import torch
from scipy.stats import chi2_contingency
from sklearn.metrics import confusion_matrix

from logger import Logger
from metrics.Metric import Metric
from utils.singleton_meta import SingletonMeta

logger = Logger().logger(__name__)


class MoEMetricPreprocessor(Metric, metaclass=SingletonMeta):
    def __init__(self, ):
        super().__init__()
        self.reset()

    def reset(self):
        self.gates = torch.tensor([], dtype=torch.long)
        self.model_index = torch.tensor([], dtype=torch.long)
        self.model_output_class = torch.tensor([], dtype=torch.long)
        self.correct = torch.tensor([], dtype=torch.long)
        self.random_correct = torch.tensor([], dtype=torch.long)
        self.labels = torch.tensor([], dtype=torch.long)
        self.super_classes = torch.tensor([], dtype=torch.long)
        self.num_experts = 0
        self.class_names = []
        self.super_classes_names = []

    def __call__(self, *args, **kwargs):
        y_pred, y_true, routes, counts, model, x, sc = self._parse_args(*args, **kwargs)
        random_routes = torch.randint(0, model.num_experts, (x.shape[0],), device=model.device)
        random_outputs = model.get_experts_output_from_indexes_list(model.get_indexes_list(random_routes),
                                                                    model.encoder(x)).argmax(
            dim=-1).cpu()
        self.gates = torch.cat((self.gates, routes), dim=0)
        self.model_index = torch.cat((self.model_index, routes), dim=0)
        self.correct = torch.cat((self.correct, (y_pred == y_true).long()), dim=0)
        self.random_correct = torch.cat((self.random_correct, (random_outputs == y_true).long()), dim=0)
        self.model_output_class = torch.cat((self.model_output_class, y_pred), dim=0)
        self.labels = torch.cat((self.labels, y_true), dim=0)
        if sc is not None:
            self.super_classes = torch.cat((self.super_classes, sc), dim=0)
        self.num_experts = model.num_experts
        self.class_names = model.train_set.classes
        self.super_classes_names = model.train_set.superclasses

    def _parse_args(self, *args, **kwargs):
        y_pred = self._get_attribute_from_args(self.possible_y_pred, **kwargs).argmax(dim=-1)
        y_true = self._get_attribute_from_args(self.possible_y_true, **kwargs)
        counts = self._get_attribute_from_args(self.possible_counts, **kwargs)
        model = self._get_attribute_from_args(self.possible_model, **kwargs)
        x = self._get_attribute_from_args(self.possible_x, **kwargs)
        routes = self._get_routes(**kwargs)
        super_classes = self._get_super_classes(**kwargs)
        return y_pred.cpu(), y_true.cpu(), routes.cpu(), counts.cpu(), model, x, super_classes.cpu()

    def _get_attribute_from_args(self, possible_args, **kwargs):
        for possible_args in possible_args:
            if possible_args in kwargs.keys():
                return kwargs[possible_args]
        raise StopIteration

    def _get_super_classes(self, **kwargs):
        try:
            return self._get_attribute_from_args(self.possible_super_classes, **kwargs)
        except StopIteration:
            return None

    def _get_routes(self, **kwargs):
        try:
            return self._get_attribute_from_args(self.possible_routes, **kwargs)
        except StopIteration:
            return self._get_attribute_from_args(self.possible_routes_probs, **kwargs).argmax(dim=-1)


class MOEMetric(Metric, metaclass=SingletonMeta):
    def __init__(self, ):
        super().__init__()
        self.moe_preprocessor = MoEMetricPreprocessor()
        self.reset()

    def reset(self):
        self.moe_preprocessor.reset()

    def __call__(self, *args, **kwargs):
        self.moe_preprocessor(*args, **kwargs)

    @property
    def gates(self):
        return self.moe_preprocessor.gates

    @property
    def model_index(self):
        return self.moe_preprocessor.model_index

    @property
    def model_output_class(self):
        return self.moe_preprocessor.model_output_class

    @property
    def correct(self):
        return self.moe_preprocessor.correct

    @property
    def random_correct(self):
        return self.moe_preprocessor.random_correct

    @property
    def labels(self):
        return self.moe_preprocessor.labels

    @property
    def super_classes(self):
        return self.moe_preprocessor.super_classes

    @property
    def num_experts(self):
        return self.moe_preprocessor.num_experts

    @property
    def class_names(self):
        return self.moe_preprocessor.class_names

    @property
    def super_classes_names(self):
        return self.moe_preprocessor.super_classes_names


class RouterVSRandomAcc(MOEMetric):
    def compute(self):
        diff = self.correct.mean(dtype=float) - self.random_correct.mean(dtype=float)
        diff_mean, diff_std = diff.mean().item(), diff.std().item()
        return torch.Tensor([diff_mean, diff_std])


class MOEConfusionMatrix(MOEMetric):
    def compute(self):
        return pd.DataFrame(confusion_matrix(self.labels, self.gates)[:, :self.num_experts], index=self.class_names)


class SuperClassConfusionMatrix(MOEMetric):
    def compute(self):
        if len(self.super_classes) == 0:
            return -1
        return pd.DataFrame(confusion_matrix(self.super_classes, self.gates)[:, :self.num_experts],
                            index=self.super_classes_names)


class PValue(MOEMetric):
    def compute(self):
        conmat = confusion_matrix(self.labels, self.gates)[:, :self.num_experts]
        try:
            return chi2_contingency(conmat)[1]
        except Exception as e:
            logger.info('p-value calculation failed with error: ', e)
            logger.debug('adding small noise to the confusion matrix')
            return chi2_contingency(conmat + np.random.uniform(1e-10, 1e-5, conmat.shape))[1]


class ExpertEntropy(MOEMetric):
    def compute(self):
        if len(self.super_classes) == 0:
            return -1
        return self.calc_entropy(np.array(self.labels))

    def calc_entropy(self, labels: np.ndarray) -> np.ndarray:
        """Computes entropy of 0-1 vector."""
        gates = np.array(self.gates)
        # calc the empirical probability of p(g_i|l_j)
        p_g_l = np.zeros((self.num_experts, len(np.unique(labels))))
        for i, l in enumerate(np.unique(labels)):
            for j, g in enumerate(np.unique(gates)):
                p_g_l[j, i] = np.mean(gates[labels == l] == g)

        # calc the entropy of each expert
        experts_entropy = np.zeros(self.num_experts)
        for i in range(self.num_experts):
            experts_entropy[i] = -np.sum(p_g_l[i] * np.log2(p_g_l[i] + 1e-10))
        max_entropy = -np.sum(1 / self.num_experts) * np.log2(1 / self.num_experts) * len(np.unique(labels))
        experts_entropy = experts_entropy / max_entropy
        return experts_entropy


class SuperClassEntropy(ExpertEntropy):
    def compute(self):
        if self.super_classes[0] == -1:
            return -1
        return self.calc_entropy(np.array(self.super_classes))


class ExpertAccuracy(MOEMetric):
    def compute(self):
        if len(self.super_classes) == 0:
            return -1
        return sum([1 for p, t in zip(self.super_classes, self.gates) if p == t]) / len(self.super_classes)


class Consistency(MOEMetric):
    def compute(self):
        consistency = np.zeros((self.num_experts, len(self.class_names)))
        for i in range(len(self.labels)):
            consistency[self.gates[i], self.labels[i]] += 1
        prob_consistency = consistency / np.maximum(consistency.sum(axis=0, keepdims=True), 1)
        entropy = -np.sum(prob_consistency * np.log2(prob_consistency + 1e-10), axis=1)
        normalized_entropy = entropy / np.log2(self.num_experts)
        return 1 - normalized_entropy


class Specialization(MOEMetric):

    def compute(self):
        specialization = np.zeros((self.num_experts, len(self.class_names)))
        total_assignments = np.zeros_like(specialization)
        for i in range(len(self.labels)):
            specialization[self.gates[i], self.labels[i]] += self.correct[i]
            total_assignments[self.gates[i], self.labels[i]] += 1
        assert total_assignments.sum() == len(self.labels)

        specialization /= np.maximum(total_assignments, 1)
        return specialization
