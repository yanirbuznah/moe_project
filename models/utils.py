import torch
import torchvision
from torch import nn

from agents import dqn, ppo
from agents.custom_env import CustomEnv
from losses import CrossEntropyLoss, MSELoss, L1Loss, SwitchLoadBalancingLoss, LossWrapper, SpecializationLoss, \
    ConsistencyLoss, RankCorrelationLoss, RegretBaseLoss, CrossEntropyRoutingLoss, FakeCrossEntropyLoss, \
    HingeBalancingLoss
from metrics.MetricsFactory import MetricsFactory
from models.MLP import MLP
from models.MOE import MixtureOfExperts


def get_agent(model: MixtureOfExperts):
    router_type = model.router_config['model_config']['model']
    if router_type.lower() == 'dqn':
        router = dqn.Agent(model)
    elif router_type.lower() == 'ppo':
        router = ppo.Agent(model)
    else:
        raise NotImplementedError(f"RL Router type {router_type} is not implemented")
    return router


def get_router(model: MixtureOfExperts):
    if 'rl' in model.router_config['type'].lower():
        return get_agent(model)
    else:
        return get_model(model.router_config['model_config'], output_shape=model.num_experts)


def get_output_shape(train_set=None, output_shape=None):
    if train_set is not None:
        return train_set.get_number_of_classes()
    elif output_shape is not None:
        return output_shape
    else:
        raise ValueError("Either train_set or output_shape must be provided")


def get_model(config: dict, *, train_set=None, output_shape=None):
    if config is None:
        return nn.Identity()
    model_config = config['model'] if 'model' in config.keys() else config
    output_shape = get_output_shape(train_set, output_shape)
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
    elif model_config['type'] == 'vit_b_16':
        model = torchvision.models.vit_b_16(num_classes=output_shape, image_size=32)
    elif model_config['type'] == 'mlp':
        model = MLP(config=model_config, output_size=output_shape)
    elif 'moe' in model_config['type'].lower():
        model = MixtureOfExperts(config=config, output_size=output_shape, train_set=train_set)
    elif model_config['type'] == 'dqn':
        pass
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
    for k, v in loss.items():
        if v is None:
            raise ValueError(f"Loss {k} is required")
    if v['type'] == 'LossCombination':
        operator = v['operator']
        losses_list = [get_loss(l) for l in v['losses']]
        weights_dict = v.get('weights', None)
        return LossWrapper(operator, losses_list, weights_dict)
    elif v['type'].lower() == 'crossentropyloss':
        return CrossEntropyLoss(**v['params'])
    elif v['type'].lower() == 'mseloss':
        return MSELoss(**v['params'])
    elif v['type'].lower() == 'l1loss':
        return L1Loss(**v['params'])
    elif v['type'].lower() == 'switchloadbalancingloss':
        return SwitchLoadBalancingLoss()
    elif v['type'].lower() == 'specializationloss':
        return SpecializationLoss(**v['params'])
    elif v['type'].lower() == 'consistencyloss':
        return ConsistencyLoss()
    elif v['type'].lower() == 'rankcorrelationloss':
        return RankCorrelationLoss(**v['params'])
    elif v['type'].lower() == 'regretbaseloss':
        return RegretBaseLoss(**v['params'])
    elif v['type'].lower() == 'fakecrossentropyloss':
        return FakeCrossEntropyLoss(**v['params'])
    elif v['type'].lower() == 'crossentropyroutingloss':
        return CrossEntropyRoutingLoss(**v['params'])
    elif v['type'].lower() == 'hingebalancingloss':
        return HingeBalancingLoss(**v['params'])
    else:
        raise NotImplementedError(f"Loss {k} not implemented")


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
