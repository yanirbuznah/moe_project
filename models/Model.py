import logging
from collections import namedtuple

import numpy as np
from wandb.integration.torch.wandb_torch import torch

from .utils import *


class OptimizerWrapper(torch.optim.Optimizer):
    def __init__(self, optimizers):
        self.optimizers = optimizers

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def state_dict(self):
        return [optimizer.state_dict() for optimizer in self.optimizers]

    def load_state_dict(self, state_dict):
        for optimizer, state in zip(self.optimizers, state_dict):
            optimizer.load_state_dict(state)





class Model(nn.Module):
    def __init__(self, config: dict, test_loader=None, train_loader = None):
        super().__init__()
        self.config = config
        model_config: dict = config.get('model')
        logging.info(f'Creating model {model_config.get("name")}')
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = get_model(model_config, train_set=train_loader.dataset)
        self.optimizer_exp = get_optimizer(self.model.experts, 'SGD', 0.1)
        self.optimizer_router = get_optimizer(self.model.router, 'adam', 0.001)

        self.optimizer = OptimizerWrapper([self.optimizer_exp, self.optimizer_router])

        self.criterion = get_loss(model_config.get('loss'))
        self.criterion.model = self.model
        self.metrics = get_metrics(config.get('metrics'), train_loader.dataset.get_number_of_active_classes())
        self.alternate = self.model.alternate if hasattr(self.model, 'alternate') else False
        self.router_assignments = []

    def to(self, device):
        self.model.to(device)
        super().to(device)
        return self

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def test_set(self):
        return self._test_set

    @test_set.setter
    def test_set(self, value):
        self._test_set = value

    def forward(self, x):
        return self.model(x)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def predict(self, x):
        return self.model(x).argmax(dim=1)

    def loss(self, batch):
        x, y, _ = batch
        x, y = x.to(self.device), y.to(self.device)
        model_output = self.forward(x)
        if isinstance(model_output, torch.Tensor):
            model_output = {'y_pred': model_output}
        model_output['target'] = y
        model_output['input'] = x
        return self.criterion(**model_output)

    def evaluate(self, batch):
        x, y, y_super = batch
        x, y = x.to(self.device), y.to(self.device)
        output = self.forward(x)
        if isinstance(output, torch.Tensor):
            output = {'output': output, 'logits': output}

        kwargs = output.copy()
        kwargs['target'] = y
        kwargs['model'] = self.model
        kwargs['input'] = x
        kwargs['super_classes'] = y_super
        self.metrics(**kwargs)
        return self.criterion(**kwargs)

    def compute_metrics(self):
        return self.metrics.compute()

    def reset_metrics(self):
        self.metrics.reset()

    def reset_parameters(self, input):
        if hasattr(self.model, 'reset_parameters'):
            self.model.reset_parameters(input)

    def get_loss_repr(self):
        return self.criterion.__repr__()

    def get_losses_details(self):
        if isinstance(self.criterion, LossWrapper):
            return self.criterion.losses
        return [self.criterion]

    def alternate_training_modules(self, router_phase):
        self.model.alternate_training_modules(router_phase)

    def router_accumulator(self):
        def calculate_entropy(tosses):
            counts = np.unique(tosses, return_counts=True)[1]
            probabilities = counts / len(tosses)
            return -np.sum(probabilities * np.log2(probabilities))

        if self.test_loader:
            batch = next(iter(self.test_loader))
            routes = self.model.router(batch[0].to(self.device)).argmax(dim=1).cpu()
            self.router_assignments.append(routes)
            if len(self.router_assignments) > 10:
                self.router_assignments.pop(0)
            # calc entropy of each raw
            if len(self.router_assignments) > 1:
                assignments = np.array(self.router_assignments)
                entropy_per_sample = np.apply_along_axis(calculate_entropy, 0, assignments)
                return namedtuple("entropy", ["stats", "mean", "std"])(entropy_per_sample, np.mean(entropy_per_sample),
                                                                       np.std(entropy_per_sample))
            return None
