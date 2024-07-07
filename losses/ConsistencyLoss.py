import torch.nn.functional

from losses import Loss


class ConsistencyLoss(Loss):
    temperature = 0.0001  # to make the gumbel softmax more deterministic

    def __init__(self):
        super().__init__()

    def __call__(self, *args, **kwargs):
        route_probabilities = next(
            kwargs[route_probabilities] for route_probabilities in self.possible_route_probabilities if
            route_probabilities in kwargs.keys())
        labels = next(kwargs[labels] for labels in self.possible_y_true if labels in kwargs.keys())
        # labels = torch.nn.functional.one_hot(labels)
        self._calc(route_probabilities, labels)
        return self.stat

    def _calc(self, route_probabilities: torch.Tensor, labels: torch.Tensor):
        one_hot_labels = torch.nn.functional.one_hot(labels)
        one_hot_routes = torch.nn.functional.gumbel_softmax(route_probabilities / self.temperature, hard=True).T
        labels_per_experts_count = one_hot_routes @ one_hot_labels.float()
        labels_per_experts_probs = (
                    labels_per_experts_count / torch.clamp(labels_per_experts_count.sum(dim=0, keepdim=True), min=1)).T
        Hx = -torch.sum(labels_per_experts_probs * torch.log2(labels_per_experts_probs + 1e-10), dim=1).mean()
        self.stat = Hx

        # gates = one_hot_routes.argmax(0)
        # gates = route_probabilities.argmax(1)
        # consistency = torch.zeros((2, 10), dtype=torch.float32).to(labels.device)
        # for i in range(len(labels)):
        #     consistency[gates[i], labels[i]] += 1
        #
        # prob_consistency = consistency / torch.clamp(consistency.sum(dim=0, keepdim=True), min=1)
        # H = -torch.sum(prob_consistency * torch.log2(prob_consistency + 1e-10))
        # max_entropy = torch.log2(torch.tensor(2, dtype=torch.float32))

        #         one_hot_labels_sum = sum(one_hot_labels).reshape(1, -1)  # vector of size (num_classes,1)
        #         probs = route_probabilities.mean(dim=0).reshape(-1, 1)
        #         consistency1 = probs * one_hot_labels_sum
        #         prob_consistency1 = consistency1 / torch.clamp(consistency1.sum(dim=1, keepdim=True), min=1)
        #         H1 = -torch.sum(prob_consistency1 * torch.log2(prob_consistency1 + 1e-10))

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
