import torch
import torchvision
import torchvision.transforms as transforms
from datasets import load_dataset
from torch import nn
from torchvision.transforms import InterpolationMode


def get_dataset(name: str, train_dataset: bool):
    if name.lower() == 'mnist':
        train = load_dataset('mnist', split='train')
        val = load_dataset('mnist', split='test')
    elif name.lower() == 'cifar10':
        train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
        val = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    elif name.lower() == 'cifar100':
        train = torchvision.datasets.CIFAR100(root='./data', train=True, download=True)
        val = torchvision.datasets.CIFAR100(root='./data', train=False, download=True)
    elif name.lower() == 'tinyimagenet':
        train = load_dataset('Maysee/tiny-imagenet', split='train')
        val = load_dataset('Maysee/tiny-imagenet', split='valid')
    elif name.lower() == 'imagenet':
        train = torchvision.datasets.ImageFolder(root='./data/imagenet/train', transform=None)
        val = torchvision.datasets.ImageFolder(root='./data/imagenet/val', transform=None)
    else:
        raise NotImplementedError(f"Dataset {name} not implemented")
    if train_dataset:
        return train
    return val


def get_transforms_from_dict(transforms_dict_config: dict, train) -> transforms.Compose:
    transforms_list = []
    for transform_name, transform_params in transforms_dict_config.items():
        if transform_name == 'ToTensor':
            transforms_list.append(transforms.ToTensor())
        elif transform_name == 'Normalize':
            transforms_list.append(transforms.Normalize(**transform_params))
        elif transform_name == 'Resize':
            transforms_list.append(transforms.Resize(**transform_params))
        elif not train:  # TODO: Check if this is the right way to do this.
            continue  # Skip all the augmentations if not training, except for ToTensor and Normalize.
        elif transform_name == 'augmentations':
            transforms_list += get_transform_from_list(transform_params['transforms'])
        else:
            transforms_list.append(getattr(transforms, transform_name)(**transform_params))

    return transforms.Compose(transforms_list)


def get_transform_from_list(transforms_list_config: list) -> list:
    transforms_list = []
    for transform in transforms_list_config:
        if transform['type'] == 'ToTensor':
            transforms_list.append(transforms.ToTensor())
        elif transform['type'] == 'Normalize':
            transforms_list.append(transforms.Normalize(transform['mean'], transform['std']))
        elif transform['type'] == 'RandomCrop':
            transforms_list.append(
                transforms.RandomCrop(transform['size'], transform['padding'], transform['padding_mode']))
        elif transform['type'] == 'RandomHorizontalFlip':
            transforms_list.append(transforms.RandomHorizontalFlip(transform['p']))
        elif transform['type'] == 'RandomVerticalFlip':
            transforms_list.append(transforms.RandomVerticalFlip(transform['p']))
        elif transform['type'] == 'RandomRotation':
            transforms_list.append(
                transforms.RandomRotation(degrees=transform['degrees'],
                                          interpolation=InterpolationMode(transform['interpolation']),
                                          expand=transform['expand'], center=transform['center'],
                                          fill=transform['fill']))
        elif transform['type'] == 'RandomResizedCrop':
            transforms_list.append(
                transforms.RandomResizedCrop(transform['size'], transform['scale'], transform['ratio'],
                                             InterpolationMode(transform['interpolation'])))
        elif transform['type'] == 'ColorJitter':
            transforms_list.append(
                transforms.ColorJitter(transform['brightness'], transform['contrast'], transform['saturation'],
                                       transform['hue']))
        elif transform['type'] == 'RandomGrayscale':
            transforms_list.append(transforms.RandomGrayscale(transform['p']))
        else:
            raise NotImplementedError(f"Transform {transform['type']} not implemented")
    return transforms_list


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
