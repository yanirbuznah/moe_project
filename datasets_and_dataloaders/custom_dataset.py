import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from datasets_and_dataloaders import utils as dutils


class CustomDataset(Dataset):
    def __init__(self, config: dict, train: bool):
        self.indices = []
        self.dataset = dutils.get_dataset(config['dataset'], train)
        self.dataset_name = config['dataset']
        self.transform = dutils.get_transforms_from_dict(config['transforms'], train) if config['transform'] else None
        self.superclasses, self.superclasses_labels = [], []
        self.init_data()

        if isinstance(self.data[0], tuple):
            self.data = [x[0] for x in self.data]

        # if isinstance(self.data[0], str):
        #     self.data = [Image.open(x) for x in self.data]

        if isinstance(self.data[0], Image.Image):
            self.data = [x.convert('RGB') for x in self.data]

    def init_data(self):
        if isinstance(self.dataset, tuple):
            self.data = []
            self.labels = []
            self.classes = []
            for d in self.dataset:
                try:
                    data = d['image'].tolist() if hasattr(d['image'], 'tolist') else d['image']
                    self.data.extend(data)
                    self.classes.extend(d.features['label'].names)
                    self.labels.extend(d['label'])
                except:
                    data = d.data if hasattr(d, 'data') else d.imgs
                    data = data.tolist() if hasattr(data, 'tolist') else data
                    self.data.extend(data)
                    self.classes.extend(d.classes)
                    self.labels.extend(d.targets)
                    # if hasattr(d, 'superclasses'):
                    #     self.superclasses.extend(d.superclasses)
                    #     self.superclasses_labels.extend(d.supertargets)
                if hasattr(self.dataset, 'superclasses'):
                    self.superclasses = self.dataset.superclasses
                    self.superclasses_labels = self.dataset.supertargets
                else:

                    superclass = type(d).__name__
                    if superclass == 'Dataset':
                        superclass = "SVHN"
                    self.superclasses.append(superclass)
                    self.superclasses_labels.extend([superclass] * len(data))
            self.superclasses_labels = [self.superclasses.index(x) for x in self.superclasses_labels]
        else:
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        if isinstance(x, str):
            x = Image.open(x)
        if isinstance(x, list):
            x = transforms.ToPILImage()(np.uint8(x))
        if isinstance(x, Image.Image):
            if x.mode != 'RGB':
                x = x.convert('RGB')

        # apply the transform (if any) to the data tensor
        x = self.transform(x)

        # get the label for this sample
        label = self.labels[index]

        super_label = self.superclasses_labels[index] if len(self.superclasses_labels) > 0 else -1

        # return the original tensor, the transformed tensor, and the label
        return x, label, super_label

    def get_original_tensor(self, index):
        return self.data[index]

    def get_transformed_tensor(self, index):
        data = self.data[index]
        if isinstance(data, str):
            data = Image.open(data)
        if isinstance(data, list):
            data = transforms.ToPILImage()(np.uint8(data))
        if isinstance(data, Image.Image):
            if data.mode != 'RGB':
                data = data.convert('RGB')

        return self.transform(data)

    def get_label(self, index):
        return self.labels[index]

    def get_class(self, index):
        return self.classes[self.labels[index]]

    def get_random_sample(self):
        index = random.randint(0, len(self.data) - 1)
        return self[index]

    def get_random_mini_batch(self, batch_size):
        indices = random.sample(range(0, len(self.data)), batch_size)
        tmp = self[indices]
        X = torch.stack([self[i][0] for i in indices])
        y = torch.LongTensor([self[i][1] for i in indices])
        return (X, y)

    def get_random_mini_batch_after_transform(self, batch_size):
        indices = random.sample(range(0, len(self.data)), batch_size)
        return self.get_mini_batch_after_transform_by_indices(indices)

    def get_mini_batch_after_transform_by_indices(self, indices):
        tmp = [self[i] for i in indices]
        X, y, _ = zip(*tmp)
        X = torch.stack(X)
        y = torch.LongTensor(y)
        return (X, y)

    def get_mini_batch_after_transform_by_interval(self, *, end, start=0):
        indices = range(start, end)
        return self.get_mini_batch_after_transform_by_indices(indices)

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

    def get_number_of_active_classes(self):
        return len(set(self.labels))
