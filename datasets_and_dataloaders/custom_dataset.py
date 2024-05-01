import random

import PIL
import torch
from torch.utils.data import Dataset

from datasets_and_dataloaders import utils as dutils


class CustomDataset(Dataset):
    def __init__(self, config: dict, train: bool):
        self.dataset = dutils.get_dataset(config['dataset'], train)
        self.dataset_name = config['dataset']
        self.transform = dutils.get_transforms_from_dict(config['transforms'],train) if config['transform'] else None
        self.superclasses, self.superclasses_labels = None, None
        try:
            self.data = self.dataset['image']
            self.classes = self.dataset.features['label'].names
            self.labels = self.dataset['label']
        except:
            self.data = self.dataset.data if hasattr(self.dataset, 'data') else self.dataset.imgs
            self.classes = self.dataset.classes
            self.labels = self.dataset.targets
            if hasattr(self.dataset, 'superclasses'):
                self.superclasses = self.dataset.superclasses
                self.superclasses_labels = self.dataset.supertargets

        if isinstance(self.data[0], PIL.Image.Image):
            self.data = [x.convert('RGB') for x in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        if self.dataset_name == 'imagenet':
            return self.dataset[index]
        # apply the transform (if any) to the data tensor
        x = self.transform(self.data[index])

        # get the label for this sample
        label = self.labels[index]

        super_label = self.superclasses_labels[index] if self.superclasses_labels is not None else -1

        # return the original tensor, the transformed tensor, and the label
        return x, label, super_label

    def get_original_tensor(self, index):
        return self.data[index]

    def get_transformed_tensor(self, index):
        return self.transform(self.data[index])

    def get_label(self, index):
        return self.labels[index]

    def get_class(self, index):
        return self.classes[self.labels[index]]

    def get_random_sample(self):
        index = random.randint(0, len(self.data) - 1)
        return self[index]

    def get_random_mini_batch(self, batch_size):
        indices = random.sample(range(0, len(self.data)), batch_size)
        X = torch.stack([self[i][0] for i in indices])
        y = torch.LongTensor([self[i][1] for i in indices])
        return (X, y)

    def get_random_mini_batch_after_transform(self, batch_size):
        indices = random.sample(range(0, len(self.data)), batch_size)
        X = torch.stack([self.get_transformed_tensor(i) for i in indices])
        y = torch.LongTensor([self[i][1] for i in indices])
        return (X, y)

    def get_random_sample_by_class(self, class_name):
        class_index = self.classes.index(class_name)
        class_indices = [i for i, x in enumerate(self.labels) if x == class_index]
        index = random.choice(class_indices)
        return self[index]

    def get_random_sample_after_transform(self):
        index = random.randint(0, len(self.data) - 1)
        return self.get_transformed_tensor(index)

    def get_random_sample_before_transform(self):
        index = random.randint(0, len(self.data) - 1)
        return self.get_original_tensor(index)

    def get_input_shape(self):
        return self.get_random_sample_after_transform().shape
    def get_number_of_classes(self):
        return len(self.classes)