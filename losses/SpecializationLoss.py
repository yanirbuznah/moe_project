import numpy as np
import torch

from losses import Loss


class SpecializationLoss(Loss):
    temperature = 0.0001

    def __init__(self):
        super().__init__()

    def __call__(self, *args, **kwargs):
        route_probabilities = next(
            kwargs[route_probabilities] for route_probabilities in self.possible_route_probabilities if
            route_probabilities in kwargs.keys())
        labels = next(kwargs[labels] for labels in self.possible_y_true if labels in kwargs.keys())
        logits = kwargs['logits']
        preds = torch.nn.functional.softmax(logits / self.temperature, dim=-1)
        self._calc(route_probabilities, labels, preds)
        return self.stat

    def _calc(self, route_probabilities, labels, preds):
        one_hot_labels = torch.nn.functional.one_hot(labels)
        one_hot_routes = torch.nn.functional.gumbel_softmax(route_probabilities / self.temperature, hard=True).T
        labels_per_experts_count = one_hot_routes @ one_hot_labels.float()
        correct_per_expert_count = one_hot_routes @ (one_hot_labels * preds)
        accuracy_per_expert_count = correct_per_expert_count / torch.maximum(labels_per_experts_count,
                                                                             torch.ones_like(
                                                                                 labels_per_experts_count))  # calculate the accuracy per expert per class
        total_assignments_probs = labels_per_experts_count / labels_per_experts_count.sum(axis=0, keepdims=True)
        specialization = (accuracy_per_expert_count * total_assignments_probs).sum(axis=0).mean()
        self.stat = 1. - specialization

        # correct = preds.argmax(1) == labels
        # gates = route_probabilities.argmax(1)
        # accuracy = np.zeros((2, 10))
        # total_assignments = np.zeros_like(accuracy)
        # for i in range(len(labels)):
        #     accuracy[gates[i], labels[i]] += correct[i]  # count the correct predictions
        #     total_assignments[gates[i], labels[i]] += 1  # count the total number of predictions
        # total_assignments_probs = total_assignments / total_assignments.sum(axis=0, keepdims=True)
        # accuracy /= np.maximum(total_assignments, 1)  # calculate the accuracy per expert per class
        # specialization = (accuracy * total_assignments_probs).sum(axis=0).mean()  # calculate the specialization
