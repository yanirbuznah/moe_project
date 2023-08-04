from losses import utils
from . import Loss


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


class LossWrapper(Loss):
    def __init__(self, operator: str, losses: list, weights: list = None):
        super().__init__()
        self.operator = Operator(operator)
        self.losses = losses
        self.weights = weights if weights is not None else [1] * len(losses)

    def __call__(self, *args, **kwargs):
        self.stat = self.operator(*[weight * loss(*args, **kwargs) for weight, loss in zip(self.weights, self.losses)])
        return self.stat

    def __repr__(self):
        return f'{self.operator.__repr__()}'.join(
            [f'({weight} * {loss.__repr__()})' for weight, loss in zip(self.weights, self.losses)])

    def __str__(self):
        return self.__repr__()
