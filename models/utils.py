import torch
import torchvision
from torch import nn

from metrics.MetricsFactory import MetricsFactory
from models.MLP import MLP
from models.MOE import MixtureOfExperts


def get_model(model_config: dict, output_shape, input_shape=None):
    if model_config['type'] == 'resnet18':
        model = torchvision.models.resnet18(num_classes=output_shape)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    elif model_config['type'] == 'resnet34':
        model = torchvision.models.resnet34(num_classes=output_shape)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    elif model_config['type'] == 'resnet50':
        model = torchvision.models.resnet50(num_classes=output_shape)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    elif model_config['type'] == 'resnet101':
        model = torchvision.models.resnet101(num_classes=output_shape)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    elif model_config['type'] == 'resnet152':
        model = torchvision.models.resnet152(num_classes=output_shape)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    elif model_config['type'] == 'vitb16':
        model = torchvision.models.vit_b_16(num_classes=output_shape, image_size=32)
    elif model_config['type'] == 'mlp':
        model = MLP(config=model_config, output_size=output_shape)
    elif model_config['type'].lower() == 'mlp_moe':
        model = MixtureOfExperts(config=model_config, input_size=input_shape, output_size=output_shape)
    else:
        raise NotImplementedError(f"Model {model_config['type']} not implemented")
    return model


def get_optimizer(model: nn.Module, optimizer: str = None, lr: float = None):
    if optimizer is None or optimizer.lower() == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr) if lr is not None else torch.optim.Adam(model.parameters())
    elif optimizer.lower() == 'sgd':
        if lr is None:
            raise ValueError("Learning rate is required for SGD optimizer")
        return torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise NotImplementedError(f"Optimizer {optimizer} not implemented")


def get_loss(loss: dict):
    for key in loss.keys():
        if key.lower() == 'crossentropyloss':
            return nn.CrossEntropyLoss()
        elif key.lower() == 'mseloss':
            return nn.MSELoss()
        else:
            raise NotImplementedError(f"Loss {key} not implemented")


def get_metrics(metrics: dict, num_classes: int):
    return MetricsFactory(metrics, num_classes)


def get_activation(activation: str):
    if activation.lower() == 'relu':
        return nn.ReLU()
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU()
    elif activation.lower() == 'tanh':
        return nn.Tanh()
    elif activation.lower() == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise NotImplementedError(f"Activation {activation} not implemented")


def get_weight_initialize_technique(technique: str):
    if technique.lower() == 'xavier':
        return nn.init.xavier_uniform_
    elif technique.lower() == 'kaiming':
        return nn.init.kaiming_uniform_
    elif technique.lower() == 'normal':
        return nn.init.normal_
    else:
        raise NotImplementedError(f"Weight initialize technique {technique} not implemented")


def get_dtype(dtype: str):
    if dtype.lower() == 'float32':
        return torch.float32
    elif dtype.lower() == 'float64':
        return torch.float64
    elif dtype.lower() == 'float16':
        return torch.float16
    else:
        raise NotImplementedError(f"Data type {dtype} not implemented")
