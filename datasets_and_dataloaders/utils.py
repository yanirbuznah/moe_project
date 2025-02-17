import torch
import torchvision
import torchvision.transforms as transforms
from datasets import load_dataset
from torch import nn
from torchvision.transforms import InterpolationMode

from utils.cifar_loader import CIFAR100


def get_dataset(name: str, train_dataset: bool):
    if name.lower() == 'mnist':
        train = load_dataset('mnist', split='train')
        val = load_dataset('mnist', split='test')
    elif name.lower() == 'cifar10':
        train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
        val = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    elif name.lower() == 'cifar100':
        train = CIFAR100(root='./data', train=True, download=True)
        val = CIFAR100(root='./data', train=False, download=True)
    elif name.lower() == 'tinyimagenet':
        train = load_dataset('Maysee/tiny-imagenet', split='train')
        val = load_dataset('Maysee/tiny-imagenet', split='valid')
    elif name.lower() == 'svhn':
        train = load_dataset('SVHN', 'cropped_digits', split='train')
        val = load_dataset('SVHN', 'cropped_digits', split='test')
    elif name.lower() == 'imagenet':
        if train_dataset:
            train = load_dataset(path='/dsi/shared/ImageNet', split='train', streaming=True)
        else:
            val = load_dataset(path='/dsi/shared/ImageNet', split='val', streaming=True)
    elif name.lower() == 'cifar10_and_mnist':
        train = (get_dataset('cifar10', True), get_dataset('mnist', True))
        val = (get_dataset('cifar10', False), get_dataset('mnist', False))
    elif name.lower() == 'cifar10_and_svhn':
        train =  (get_dataset('cifar10', True), get_dataset('SVHN', True))
        val =  (get_dataset('cifar10', False), get_dataset('SVHN', False))
    else:
        raise NotImplementedError(f"Dataset {name} not implemented")
    if train_dataset:
        return train
    return val


def get_transforms_from_dict(transforms_dict_config: dict, train) -> transforms.Compose:
    transforms_list = []
    for transform_name, transform_params in transforms_dict_config.items():
        if transform_name == 'ToPILImage':
            transforms_list.append(transforms.ToPILImage())
        elif transform_name == 'ToTensor':
            transforms_list.append(transforms.ToTensor())
        elif transform_name == 'Normalize':
            transforms_list.append(transforms.Normalize(**transform_params))
        elif transform_name == 'Resize':
            transforms_list.append(transforms.Resize(**transform_params))
        elif transform_name == 'CenterCrop':
            transforms_list.append(transforms.CenterCrop(**transform_params))
        elif not train:  # TODO: Check if this is the right way to do this.
            continue  # Skip all the augmentations if not training, except for ToTensor and Normalize.
        elif transform_name == 'GaussianBlur':
            transforms_list.append(transforms.GaussianBlur(**transform_params))
        elif transform_name == 'augmentations':
            transforms_list += get_transform_from_list(transform_params['transforms'])
        else:
            transforms_list.append(getattr(transforms, transform_name)(**transform_params))

    return transforms.Compose(transforms_list)


def get_transform_from_list(transforms_list_config: list) -> list:
    transforms_list = []
    for transform in transforms_list_config:
        if transform['type'] == 'ToPILImage':
            transforms_list.append(transforms.ToPILImage())
        elif transform['type'] == 'ToTensor':
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
