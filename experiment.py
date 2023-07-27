import torch.utils.data as data

import utils.general_utils
from datasets_and_dataloaders.custom_dataset import CustomDataset
from models.Model import Model


class Experiment:

    def __init__(self, config: dict):
        self.config = config

        self.trainset = CustomDataset(config['dataloader'], train=True)
        self.testset = CustomDataset(config['dataloader'], train=False)

        self.trainloader = data.DataLoader(self.trainset, batch_size=config['dataloader']['batch_size'],
                                           shuffle=config['dataloader']['shuffle'],
                                           num_workers=config['dataloader']['num_workers'])
        self.testloader = data.DataLoader(self.testset, batch_size=config['dataloader']['batch_size'], shuffle=False,
                                          num_workers=config['dataloader']['num_workers'])

        self.classes = self.trainset.classes
        self.num_of_classes = len(self.classes)
        self.input_shape = self.trainset.get_random_sample_after_transform().shape

        self.model = Model(config, self.input_shape, self.num_of_classes).to(utils.general_utils.device)

    def run(self):
        x = utils.general_utils.run_train_epoch(self.model, self.trainloader)
        utils.general_utils.evaluate(self.model, self.testloader)
        return x
