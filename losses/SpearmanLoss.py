import torch.nn.functional

from losses import Loss


class SpearmanLoss(Loss):

    def __init__(self):
        super().__init__()

    def __call__(self, *args, **kwargs):
        route_probabilities = next(
            kwargs[route_probabilities] for route_probabilities in self.possible_route_probabilities if
            route_probabilities in kwargs.keys())
        labels = next(kwargs[labels] for labels in self.possible_y_true if labels in kwargs.keys())
        labels = torch.nn.functional.one_hot(labels)
        self._calc(route_probabilities, labels)
        return self.stat

    import torch
    import torch.nn.functional as F

    def _calc(self, route_probabilities, labels):
        gates = route_probabilities.argmax(1)
        consistency = torch.zeros((2, 10), dtype=torch.float32).to(labels.device)
        for i in range(len(labels)):
            consistency[gates[i], labels[i]] += 1
        prob_consistency = consistency / torch.clamp(consistency.sum(dim=0, keepdim=True), min=1)
        H = -torch.sum(prob_consistency * torch.log2(prob_consistency))
        max_entropy = (torch.
                       log2(torch.tensor(2, dtype=torch.float32)))
        return 1 - (H / max_entropy)

    # def _calc(self, route_probabilities, labels):
    #     # route_max, _ = torch.max(route_probabilities, dim=1)
    #     # consistency = route_max * labels.T
    #     consistency = np.zeros((num_experts, num_classes))
    #     for i in range(len(labels)):
    #         consistency[gates[i], labels[i]] += 1
    #     prob_consistency = consistency / np.maximum(consistency.sum(axis=0, keepdims=True), 1)
    #     H = entropy(prob_consistency, base=2)
    #     max_entropy = np.log2(num_experts)
    #     return 1 - (H.mean() / max_entropy)
