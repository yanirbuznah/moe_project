import torch
import torchvision
from torch import nn

from metrics.utils import MetricsFactory


def get_model(model_config: dict, input_shape, output_shape):
    if model_config['type'] == 'resnet18':
        model = torchvision.models.resnet18(num_classes=output_shape)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model
    else:
        raise NotImplementedError(f"Model {model_config['type']} not implemented")


def get_optimizer(model: nn.Module, optimizer: str = None, lr: float = None):
    if optimizer is None or optimizer.lower() == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr) if lr is not None else torch.optim.Adam(model.parameters())
    elif optimizer.lower() == 'sgd':
        if lr is None:
            raise ValueError("Learning rate is required for SGD optimizer")
        return torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise NotImplementedError(f"Optimizer {optimizer} not implemented")

def get_loss(loss:dict):
    for key in loss.keys():
        if key.lower() == 'crossentropyloss':
            return nn.CrossEntropyLoss()
        elif key.lower() == 'mseloss':
            return nn.MSELoss()
        else:
            raise NotImplementedError(f"Loss {key} not implemented")


def get_metrics(metrics:dict,num_classes:int):
    return MetricsFactory(metrics,num_classes)