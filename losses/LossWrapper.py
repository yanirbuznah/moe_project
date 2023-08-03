import torch
from torch import nn

from losses import utils


class Operator:
    def __init__(self, operator: str):
        self.operator = utils.get_operator(operator)
        self.name = operator

    def __call__(self, *args, **kwargs):
        return self.operator(*args, **kwargs)

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.__repr__()


class LossWrapper:
    def __init__(self, operator: str, losses: list):
        self.operator = Operator(operator)
        self.losses = losses

    def __call__(self, *args, **kwargs):
        return self.operator(*[loss(*args, **kwargs) for loss in self.losses])

    def __repr__(self):
        return f' {self.operator.__repr__()} '.join([f'{loss.__repr__()}' for loss in self.losses])

    def __str__(self):
        return self.__repr__()


class LossWrapperWithWeights(LossWrapper):
    def __init__(self, operator: str, losses: list, weights: list):
        super().__init__(operator, losses)
        self.weights = weights

    def __call__(self, *args, **kwargs):
        return self.operator(*[weight * loss(*args, **kwargs) for weight, loss in zip(self.weights, self.losses)])

    def __repr__(self):
        return f'{self.operator.__repr__()}'.join(
            [f'({weight} * {loss.__repr__()})' for weight, loss in zip(self.weights, self.losses)])

    def __str__(self):
        return self.__repr__()

