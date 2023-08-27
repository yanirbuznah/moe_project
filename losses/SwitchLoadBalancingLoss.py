from . import Loss


class SwitchLoadBalancingLoss(Loss):
    def __init__(self):
        super().__init__()


    def __call__(self, *args, **kwargs):
        route_probabilities = next(
            kwargs[route_probabilities] for route_probabilities in self.possible_route_probabilities if
            route_probabilities in kwargs.keys())
        counts = next(kwargs[counts] for counts in self.possible_counts if counts in kwargs.keys())
        self._calc(route_probabilities, counts)
        return self.stat

    def _calc(self, route_probabilities, counts):
        total = counts.sum(-1, keepdim=True)
        route_fraction = counts / total
        route_prob = route_probabilities.sum(0) / total
        num_of_experts = route_probabilities.shape[-1]
        self.stat = num_of_experts * (route_fraction * route_prob).sum()
