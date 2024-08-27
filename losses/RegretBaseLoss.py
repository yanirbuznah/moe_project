import torch.nn.functional

from losses import Loss


class RegretBaseLoss(Loss):

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
        # get the top k routes
        top_k_probs, topk = torch.topk(route_probabilities, self.k, dim=1)

        # get the cross entropy loss for the top k routes
        ce_losses = [torch.nn.functional.cross_entropy(logits, labels, reduction='none')]
        for i in range(1, self.k):
            logits_i = self.model.get_unsupervised_output(x, routes=topk[:, i])
            ce_loss = torch.nn.functional.cross_entropy(logits_i, labels, reduction='none')
            ce_losses.append(ce_loss)
        ce_losses = torch.stack(ce_losses, dim=1)
        if self.detach:
            ce_losses = ce_losses.detach()

        # current_score =  E[P*Loss] = \sum_i P_i * Loss_i
        top_k_probs = top_k_probs / top_k_probs.sum(dim=1, keepdim=True)
        current_score = (top_k_probs * ce_losses).sum(dim=1) #

        best_loss, _ = ce_losses.min(dim=1)

        regret = current_score - best_loss


        assert all(regret >= -1e-5), f"Regret must be non-negative, but got {regret}"

        self.stat = regret.mean()

    # check if everything is differentiable with respect to the router probabilities