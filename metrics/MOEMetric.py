from argparse import Namespace

import numpy as np
import pandas as pd
import torch
from numpy import dtype
from scipy.stats import chi2_contingency, entropy, spearmanr
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
        self.model = None
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
        self.route_probs = torch.tensor([], dtype=torch.float)
        self.input = torch.tensor([], dtype=torch.float)

    def __call__(self, *args, **kwargs):
        y_pred, y_true, routes, counts, model, x, sc, route_probs = self._parse_args(*args, **kwargs)
        self.num_experts = model.num_experts
        random_routes = (routes + 1) % self.num_experts  # torch.randint(0, model.num_experts, (x.shape[0],), device=model.device)
        random_outputs = model.get_experts_logits_from_indexes_list(model.get_indexes_list(random_routes),
                                                                    model.encoder(x)).argmax(dim=-1).cpu()
        self.route_probs = torch.cat((self.route_probs, route_probs), dim=0) if route_probs is not None else None
        self.gates = torch.cat((self.gates, routes), dim=0)
        self.model_index = torch.cat((self.model_index, routes), dim=0)
        self.correct = torch.cat((self.correct, (y_pred == y_true).long()), dim=0)
        self.random_correct = torch.cat((self.random_correct, (random_outputs == y_true).long()), dim=0)
        self.model_output_class = torch.cat((self.model_output_class, y_pred), dim=0)
        self.labels = torch.cat((self.labels, y_true), dim=0)
        self.input = torch.cat((self.input, x.cpu()), dim=0)
        self.super_classes = torch.cat((self.super_classes, sc), dim=0) if sc is not None else None
        self.class_names = model.train_set.classes
        self.super_classes_names = model.train_set.superclasses
        self.model = model

    def _parse_args(self, *args, **kwargs):
        y_pred = self._get_attribute_from_args(self.possible_y_pred, **kwargs).argmax(dim=-1)
        y_true = self._get_attribute_from_args(self.possible_y_true, **kwargs)
        counts = self._get_attribute_from_args(self.possible_counts, **kwargs)
        model = self._get_attribute_from_args(self.possible_model, to_cpu=False, **kwargs)
        x = self._get_attribute_from_args(self.possible_x, to_cpu=False, **kwargs)
        routes = self._get_routes(**kwargs)
        super_classes = self._get_attribute_from_args(self.possible_super_classes, **kwargs)
        route_probs = self._get_attribute_from_args(self.possible_routes_probs, **kwargs)
        return y_pred, y_true, routes, counts, model, x, super_classes, route_probs

    def _get_attribute_from_args(self, possible_args, to_cpu=True, **kwargs):
        for possible_args in possible_args:
            if possible_args in kwargs.keys():
                args = kwargs[possible_args]
                return args.cpu() if to_cpu else args
        return None

    def _get_routes(self, to_cpu=True, **kwargs):
        try:
            return self._get_attribute_from_args(self.possible_routes, to_cpu, **kwargs)
        except StopIteration:
            return self._get_attribute_from_args(self.possible_routes_probs, to_cpu, **kwargs).argmax(dim=-1)


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

    @property
    def route_probs(self):
        return self.moe_preprocessor.route_probs

    @property
    def model(self):
        return self.moe_preprocessor.model

    @property
    def input(self):
        return self.moe_preprocessor.input


class RouterVSRandomAcc(MOEMetric):
    def compute(self):
        diff = self.correct.mean(dtype=float) - self.random_correct.mean(dtype=float)
        diff_mean, diff_std = diff.mean().item(), diff.std().item()
        return torch.Tensor([diff_mean, diff_std])


class MOEConfusionMatrix(MOEMetric):
    def compute(self):
        return pd.DataFrame(confusion_matrix(self.labels, self.gates)[:, :self.num_experts], index=self.class_names)

    @staticmethod
    def compute_manual(*, gates, labels, num_experts, class_names):
        return pd.DataFrame(confusion_matrix(labels, gates)[:, :num_experts], index=class_names)


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
        prob_consistency = consistency / np.maximum(consistency.sum(axis=0, keepdims=True), 1)  # pro

        H = entropy(prob_consistency, base=2)  #
        max_entropy = np.log2(self.num_experts)
        return 1 - (H.mean() / max_entropy)

    @staticmethod
    def compute_manual(*, gates, labels, num_experts, num_classes):
        consistency = np.zeros((num_experts, num_classes))
        for i in range(len(labels)):
            consistency[gates[i], labels[i]] += 1
        prob_consistency = consistency / np.maximum(consistency.sum(axis=0, keepdims=True), 1)
        H = entropy(prob_consistency, base=2)
        max_entropy = np.log2(num_experts)
        return 1 - (H.mean() / max_entropy)


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

    @staticmethod
    def compute_manual(*, gates, labels, correct, num_experts, num_classes):
        specialization = np.zeros((num_experts, num_classes))
        total_assignments = np.zeros_like(specialization)
        for i in range(len(labels)):
            specialization[gates[i], labels[i]] += correct[i]  # count the correct predictions
            total_assignments[gates[i], labels[i]] += 1  # count the total number of predictions
        assert total_assignments.sum() == len(labels)

        specialization /= np.maximum(total_assignments, 1)  # calculate the accuracy per expert per class
        return specialization


class NewSpecialization(MOEMetric):

    def compute(self):
        accuracy = np.zeros((self.num_experts, len(self.class_names)))
        total_assignments = np.zeros_like(accuracy)
        for i in range(len(self.labels)):
            accuracy[self.gates[i], self.labels[i]] += self.correct[i]
            total_assignments[self.gates[i], self.labels[i]] += 1
        total_assignments_probs = total_assignments / total_assignments.sum(axis=0, keepdims=True)
        accuracy /= np.maximum(total_assignments, 1)  # calculate the accuracy per expert per class
        specialization = (accuracy * total_assignments_probs).sum(axis=0).mean()  # calculate the specialization
        return specialization

    @staticmethod
    def compute_manual(*, gates, labels, correct, num_experts, num_classes):
        accuracy = np.zeros((num_experts, num_classes))
        total_assignments = np.zeros_like(accuracy)
        for i in range(len(labels)):
            accuracy[gates[i], labels[i]] += correct[i]  # count the correct predictions
            total_assignments[gates[i], labels[i]] += 1  # count the total number of predictions
        total_assignments_probs = total_assignments / total_assignments.sum(axis=0, keepdims=True)
        accuracy /= np.maximum(total_assignments, 1)  # calculate the accuracy per expert per class
        specialization = (accuracy * total_assignments_probs).sum(axis=0).mean()  # calculate the specialization
        return specialization


class DeadExperts(MOEMetric):
    def compute(self):
        return self.num_experts - self.gates.unique().shape[0]

    @staticmethod
    def compute_manual(*, gates):
        return gates.unique().shape[0]


class AccuracyDiff(MOEMetric):
    def compute(self):
        """
        Check the Accuracy difference between the routed experts and random experts
        """
        diff = self.correct.mean(dtype=float) - self.random_correct.mean(dtype=float)
        return diff.item()


class SpearmanCorrelation(MOEMetric):

    def get_logits(self, input_batches, route_indices_batches, i):
        logits = []
        for x, routes in zip(input_batches, route_indices_batches):
            x = x.to(self.model.device)
            logits.append(self.model.get_unsupervised_output(x, routes=torch.zeros_like(routes[:, 0]) + i))
        return torch.cat(logits)

    def compute(self):
        if self.route_probs is None:
            return -1
        route_probabilities_sorted, route_indices = torch.sort(self.route_probs, dim=1, descending=True)
        route_indices = self.route_probs.argsort(dim=1, descending=True)
        # route_probabilities_sorted = route_probabilities_sorted[:, :self.num_experts]
        input_batches = torch.split(self.input, 1000)
        route_indices_batches = torch.split(route_indices, 1000)
        # get the cross entropy loss for the top k routes
        ce_losses = []
        with torch.no_grad():
            for i in range(self.num_experts):
                # eval by batches
                logits_i = self.get_logits(input_batches, route_indices_batches, i)
                ce_loss = torch.nn.functional.cross_entropy(logits_i, self.labels.to(self.model.device),
                                                            reduction='none')
                ce_losses.append(ce_loss)
            ce_losses = torch.stack(ce_losses, dim=1)

        # calculate the spearman correlation between the route probabilities and the cross entropy losses
        ce_losses_rank = ce_losses.argsort(dim=1).cpu()
        return self._compute_spearman_correlation(ce_losses_rank, route_indices)

    def _compute_spearman_correlation(self, a, b):
        import numpy as np

        # Compute the differences between ranks
        d = a - b

        # Compute the squared differences
        d_squared = np.square(d)

        # Compute the Spearman correlation using the formula
        n = a.shape[1]
        spearman_correlations = 1 - (6 * torch.sum(d_squared, dim=1)) / (n * (n ** 2 - 1))

        # Compute the mean Spearman correlation
        return round(torch.mean(spearman_correlations).item(), 5)


if __name__ == '__main__':
    import pickle


    def convert_args(x):
        x = {'gates': x['routes'], 'labels': x['target'], 'correct': x['target'] == torch.argmax(x['output'], dim=-1),
             'num_experts': len(x['counts']), 'num_classes': 1 + max(x['target']).item()}
        return x


    x = pickle.load(open('/home/dsi/buznahy/moe_project/metrics/kwargs.pickle', 'rb'))
    res = DeadExperts.compute_manual(gates=x['gates'])
    confusion_matrix1 = MOEConfusionMatrix.compute_manual(
        gates=x['gates'], labels=x['labels'], num_experts=x['num_experts'],
        class_names=[str(i) for i in range(x['num_classes'])])

    res1 = Specialization.compute_manual(gates=x['gates'], labels=x['labels'], correct=x['correct'],
                                         num_experts=x['num_experts'], num_classes=x['num_classes'])

    res2 = NewSpecialization.compute_manual(gates=x['gates'], labels=x['labels'], correct=x['correct'],
                                            num_experts=x['num_experts'], num_classes=x['num_classes'])

    res3 = Consistency.compute_manual(gates=x['gates'], labels=x['labels'], num_experts=x['num_experts'],
                                      num_classes=x['num_classes'])
    print(f'Confusion Matrix:\n{confusion_matrix1}')
    print(f'Specialization: {res1}')
    print(f'New Specialization: {res2}')
    print(f'Consistency: {res3}')
    print('Expertise:', res1.mean() * res3)
    print('New Expertise:', res2 * res3)
