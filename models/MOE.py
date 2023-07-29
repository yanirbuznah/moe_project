import torch
from torch import nn

from . import utils


class MixtureOfExperts(nn.Module):
    def __init__(self, *, config, input_size, output_size):
        super().__init__()
        self.alternate = config.get('alternate_epoch', 0)
        self.num_experts = config['num_experts']
        self.k = config.get('k', 1)
        self.input_size = input_size
        self.output_size = output_size
        self.input_size_router = config.get('input_size_router', input_size)
        self.experts = nn.ModuleList(
            [utils.get_model(config['expert'], self.output_size) for _ in range(self.num_experts)])

        self.unsupervised_router = not config['router'][0]['supervised']
        self.router = utils.get_model(config['router'][0]['model_config'], self.num_experts)

        self.softmax = nn.Softmax(dim=-1)
        self.encoder = utils.get_model(config['encoder'], output_shape=self.input_size_router) if config.get('encoder',
                                                                                                             False) else nn.Identity()

    def reset_parameters(self, input):
        for expert in self.experts:
            expert.reset_parameters(input)
        self.router.reset_parameters(input)

    def router_phase(self, router_phase):
        self._alternate_modules(router_phase)

    def unsupervised_router_step(self, x):
        return self.unsupervised_router.act(x)  # .cpu())
        # p = self.unsupervised_router.policy.get_probs(a)
        # return p

    def supervised_router_step(self, x, router_phase):
        # encode the input to linear space
        x_enc = self.encoder(x)

        if not self.unsupervised_router and self.alternate and self.training:
            self._alternate_modules(router_phase)

        return self.router(x_enc)

    def forward_unsupervised(self, x, *, routes=None):
        # get the routing probabilities
        routes = self.unsupervised_router_step(x) if routes is None else routes

        indexes_list = self.get_indexes_list(routes)

        counts = x.new_tensor([len(indexes_list[i]) for i in range(self.num_experts)])

        output = self.get_experts_output_from_indexes_list(indexes_list, x)

        return {'output': output, 'routes': routes, 'counts': counts}

    def forward_supervised(self, x, *, router_phase=False):
        # get the routing probabilities
        router_probs = self.supervised_router_step(x, router_phase)

        # router_probs = self.softmax(router_output)
        router_probs_max, routes = torch.max(router_probs, dim=-1)

        # get the indexes of the samples for each expert
        indexes_list = self.get_indexes_list(routes)

        # get the counts of the samples for each expert
        counts = x.new_tensor([len(indexes_list[i]) for i in range(self.num_experts)])

        # get the output of the experts
        output = self.get_experts_output_from_indexes_list(indexes_list, x)
        output = output * (router_probs_max / router_probs_max.detach()).view(-1, 1)

        return {'output': output, 'router_probs': router_probs, 'counts': counts}

    def forward(self, x, *, router_phase=False):

        if self.unsupervised_router:
            return self.forward_unsupervised(x)
        else:
            return self.forward_supervised(x, router_phase=router_phase)

    def encode(self, x):
        if self.unsupervised_router:
            return self.unsupervised_router.q_net.encoder(x)
        return self.encoder(x)

    def get_experts_output_from_indexes_list(self, indexes_list, x):
        experts_output = []
        for i in range(self.num_experts):
            if len(indexes_list[i]) > 0:
                experts_output.append(self.experts[i](x[indexes_list[i], :]))
            else:
                experts_output.append(x.new_zeros((0, self.output_size)))

        indexes_list = torch.cat(indexes_list, dim=0).to(x.device)
        experts_output = torch.cat(experts_output, dim=0).to(x.device)

        output = x.new_zeros(experts_output.shape)
        output.index_copy_(0, indexes_list, experts_output)

        return output

    def get_indexes_list(self, routes):
        return [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.num_experts)]

    def _alternate_modules(self, router_phase):
        if router_phase:
            self.router.requires_grad_(True)
            self.encoder.requires_grad_(True)
            for expert in self.experts:
                expert.requires_grad_(False)
        else:
            self.router.requires_grad_(False)
            self.encoder.requires_grad_(False)
            for expert in self.experts:
                expert.requires_grad_(True)
