import torch.nn.functional

from losses import Loss


class RankCorrelationLoss(Loss):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.k = kwargs.get('k', 2)
        self.detach = kwargs.get('detach_experts_grad', False)

    def __call__(self, *args, **kwargs):
        route_probabilities = next(
            kwargs[route_probabilities] for route_probabilities in self.possible_route_probabilities if
            route_probabilities in kwargs.keys())
        labels = next(kwargs[labels] for labels in self.possible_y_true if labels in kwargs.keys())
        x = kwargs['input']
        logits = kwargs['logits']
        self._calc(route_probabilities, labels, x, logits)
        return self.stat

    def _calc(self, route_probabilities, labels, x, logits):
        route_probabilities_sorted, route_indices = torch.sort(route_probabilities, dim=1, descending=True)
        route_probabilities_sorted = route_probabilities_sorted[:, :self.k]
        # get the cross entropy loss for the top k routes
        ce_losses = []
        for i in range(self.k):
            logits_i = self.model.get_unsupervised_output(x, routes=route_indices[:, i])
            ce_loss = torch.nn.functional.cross_entropy(logits_i, labels, reduction='none')
            ce_losses.append(ce_loss)
        ce_losses = -1 * torch.stack(ce_losses, dim=1)
        # if self.detach:
        #     ce_losses = ce_losses.detach()
        ce_losses_normalized = ce_losses - torch.mean(ce_losses, dim=1, keepdim=True) # X - E[X]

        route_probabilities_normalized = route_probabilities - torch.mean(route_probabilities, dim=1,
                                                                                 keepdim=True) # Y - E[Y]

        # find the covariance between route_probabilities and ce_losses
        cov = torch.sum(route_probabilities_normalized * ce_losses_normalized, dim=1)
        # find the standard deviation of the route_probabilities
        route_std = torch.sqrt(torch.sum(route_probabilities_normalized ** 2, dim=1))
        # find the standard deviation of the ce_losses
        ce_std = torch.sqrt(torch.sum(ce_losses_normalized ** 2, dim=1))
        # find the pearson correlation
        correlation = cov / ((route_std * ce_std) + 1e-8)
        self.stat = 1 - correlation.mean()

        # # sort the losses from the lowest to the highest, and get the indices
        # _, indices = torch.sort(ce_losses, dim=1)
        # # sort the route probabilities from the highest to the lowest, and get the indices
        # _, route_indices = torch.sort(route_probabilities, dim=1, descending=True)

    def _spearman_correlation(self, indices, route_indices):
        n = indices.size(0)
        # calculate the spearman correlation
        correlation = 1 - 6 * (torch.sum((indices - route_indices) ** 2) / (n * (n ** 2 - 1)))
        return correlation
