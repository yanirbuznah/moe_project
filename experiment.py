import json
import logging
import os

import pandas as pd
import torch.utils.data as data

import utils.general_utils as utils
from datasets_and_dataloaders.custom_dataset import CustomDataset
from models.MOE import MixtureOfExperts
from models.Model import Model

logger = logging.getLogger(__name__)


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Experiment(metaclass=SingletonMeta):
    def __init__(self, config: dict = None):
        if hasattr(self, 'initialized'):
            return

        self.config = config
        self.train_set = CustomDataset(config['dataloader'], train=True)
        self.test_set = CustomDataset(config['dataloader'], train=False)

        self.train_loader = data.DataLoader(self.train_set, batch_size=config['dataloader']['batch_size'],
                                            shuffle=config['dataloader']['shuffle'],
                                            num_workers=config['dataloader']['num_workers'])
        self.test_loader = data.DataLoader(self.test_set, batch_size=config['dataloader']['batch_size'], shuffle=False,
                                           num_workers=config['dataloader']['num_workers'])

        self.classes = self.train_set.classes
        self.num_of_classes = len(self.classes)
        dummy_sample = self.train_set.get_random_sample_after_transform()

        self.model = Model(config, self.num_of_classes, self.train_set).to(utils.device)
        self.model.reset_parameters(dummy_sample.view(1, *dummy_sample.shape).to(utils.device))
        self._init_experiment_folder()

        self.initialized = True

    def __repr__(self):
        return f"Experiment: {self.config['log']['experiment_name']}"

    def _init_experiment_folder(self):
        self.experiment_path = utils.get_experiment_path(self.config.get('log').get('experiment_name'))
        if not os.path.exists(self.experiment_path):
            os.makedirs(self.experiment_path)
        json.dump(self.config, open(os.path.join(self.experiment_path, "config.json"), 'w'), indent=4)

    def run_rl_combined_model(self, epochs=None):
        epochs = epochs or self.config['epochs']
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch}")
            # todo: check alternating training
            self.model.model.train_router(500)
            utils.run_train_epoch(self.model, self.train_loader)
            train_eval_result = utils.evaluate(self.model, self.train_loader)
            print(train_eval_result)
            evaluate_result = utils.evaluate(self.model, self.test_loader)
            self.save_results_in_experiment_folder(epoch, evaluate_result)

    def run_normal_model(self, epochs=None):
        epochs = epochs or self.config['epochs']
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch}")
            utils.run_train_epoch(self.model, self.train_loader)
            train_eval_result = utils.evaluate(self.model, self.train_loader)
            evaluate_result = utils.evaluate(self.model, self.test_loader)
            print(evaluate_result)
            self.save_results_in_experiment_folder(epoch, evaluate_result)

    def run(self):
        model = self.model.model
        if isinstance(model, MixtureOfExperts):
            # TODO: add epochs by type
            if model.unsupervised_router:
                self.run_rl_combined_model(self.model.config['epochs'])
            else:
                self.run_normal_model(self.model.config['epochs'])
        else:
            self.run_normal_model()


    def save_results_in_experiment_folder(self, epoch, evaluate_result):
        results_csv = os.path.join(self.experiment_path, "results.csv")
        df = pd.DataFrame.from_dict(evaluate_result, orient='index').T
        df.insert(0, 'epoch', epoch)
        df.to_csv(results_csv, mode='a', header=not os.path.exists(results_csv), index=False)
