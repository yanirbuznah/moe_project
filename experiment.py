import json
import logging
import os

import pandas as pd
import torch.utils.data as data

import utils.general_utils as utils
from datasets_and_dataloaders.custom_dataset import CustomDataset
from models.Model import Model

logger = logging.getLogger(__name__)


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
        dummy_sample = self.trainset.get_random_sample_after_transform()
        self.input_shape = dummy_sample.shape

        self.model = Model(config, self.input_shape, self.num_of_classes).to(utils.device)
        self.model.reset_parameters(dummy_sample.view(1, -1).to(utils.device))
        self._init_experiment_folder()

    def _init_experiment_folder(self):
        self.experiment_path = utils.get_experiment_path(self.config.get('log').get('experiment_name'))
        if not os.path.exists(self.experiment_path):
            os.makedirs(self.experiment_path)
        json.dump(self.config, open(os.path.join(self.experiment_path, "config.json"), 'w'), indent=4)

    def run(self):
        for epoch in range(self.config['epochs']):
            logger.info(f"Epoch {epoch}")
            utils.run_train_epoch(self.model, self.trainloader)
            train_eval_result = utils.evaluate(self.model, self.trainloader)
            print(train_eval_result)
            evaluate_result = utils.evaluate(self.model, self.testloader)
            self.save_results_in_experiment_folder(epoch, evaluate_result)

    def save_results_in_experiment_folder(self, epoch, evaluate_result):
        results_csv = os.path.join(self.experiment_path, "results.csv")
        df = pd.DataFrame.from_dict(evaluate_result, orient='index').T
        df.insert(0, 'epoch', epoch)
        df.to_csv(results_csv, mode='a', header=not os.path.exists(results_csv), index=False)