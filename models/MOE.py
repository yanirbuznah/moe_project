import logging
import traceback

import torch
from torch import nn

from . import utils


class MixtureOfExperts(nn.Module):
    def __init__(self, *, config, output_size, train_set=None):
        super().__init__()
        self.config = config
        model_config = config['model']
        self.train_set = train_set
        self.input_shape = train_set.get_input_shape()
        self.alternate = model_config.get('alternate', 0)
        self.num_experts = model_config['num_experts']
        self.k = model_config.get('k', 1)
        self.output_shape = output_size
        self.input_shape_router = model_config.get('input_size_router', self.input_shape)
        self.experts = nn.ModuleList(
            [utils.get_model(model_config['expert'], train_set=train_set) for _ in range(self.num_experts)])

        self.unsupervised_router = not model_config['router'][0]['supervised']
        self.router_config = model_config['router'][0]
        self.router = utils.get_router(self)

        self.softmax = nn.Softmax(dim=-1)
        self.encoder = utils.get_model(model_config['encoder'],
                                       output_shape=self.input_shape_router) if model_config.get('encoder',
                                                                                                 False) else nn.Identity()

    def to(self, device):
        self.experts.to(device)
        self.router.to(device)
        self.encoder.to(device)
        super().to(device)
        return self

    @property
    def device(self):
        return next(self.parameters()).device

    def reset_parameters(self, input):
        for expert in self.experts:
            if hasattr(expert, 'reset_parameters'):
                expert.reset_parameters(input)
        if hasattr(self.router, 'reset_parameters'):
            self.router.reset_parameters(input)

    def router_phase(self, router_phase):
        self._alternate_modules(router_phase)

    def unsupervised_router_step(self, x):
        return self.router.act(x)  # .cpu())

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

    def get_unsupervised_output(self, x, *, routes=None):
        routes = self.unsupervised_router_step(x) if routes is None else routes

        indexes_list = self.get_indexes_list(routes)

        return self.get_experts_output_from_indexes_list(indexes_list, x)

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
                experts_output.append(x.new_zeros((0, self.output_shape)))

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

    def init_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                self.wight_init(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _reset_parameters(self, model, input):
        self.forward(input)
        for layer in model.layers:
            if isinstance(layer, nn.Linear):
                self.wight_init(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def train_router(self, epoch):
        if epoch % self.alternate == 0 and epoch > self.alternate:
            try:
                self.router.learn()
            except Exception as e:
                logging.error("Error in training router: {}".format(e))
                traceback.print_exc()
                raise e
