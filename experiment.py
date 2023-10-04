import json
import logging
import os

import pandas as pd
import torch.nn
import torch.utils.data as data

import utils.general_utils as utils
from datasets_and_dataloaders.custom_dataset import CustomDataset
from metrics import ConfusionMatrix
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
        dummy_sample = self.train_set.get_random_sample_after_transform()

        self.model = Model(config, self.train_set).to(utils.device)
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

    def evaluate_and_save_results(self, epoch: int, model: torch.nn.Module, loader: data.DataLoader,
                                  path: str = 'results.csv'):
        evaluate_result = utils.evaluate(model, loader)
        print(f"Epoch:{epoch}:\n{evaluate_result}")
        self.save_results_in_experiment_folder(epoch, evaluate_result, path=path)

    def run_rl_combined_model(self, epoch):
        logger.info(f"Epoch {epoch}")
        utils.run_train_epoch(self.model, self.train_loader)
        self.model.model.train_router(epoch)
        self.evaluate_and_save_results(epoch, self.model, self.train_loader, path='train_results.csv')
        self.evaluate_and_save_results(epoch, self.model, self.test_loader, path='test_results.csv')

    def run_normal_model(self, epoch):
        logger.info(f"Epoch {epoch}")
        utils.run_train_epoch(self.model, self.train_loader)
        self.evaluate_and_save_results(epoch, self.model, self.train_loader, path='train_results.csv')
        self.evaluate_and_save_results(epoch, self.model, self.test_loader, path='test_results.csv')

    def run(self):
        model = self.model.model
        for epoch in range(self.model.config['epochs']):
            if isinstance(model, MixtureOfExperts):
                rl_router = model.unsupervised_router
                if rl_router and self.model.model.router_config['epochs']:
                    self.run_rl_combined_model(epoch)
                else:
                    self.run_normal_model(epoch)
                if not self.model.model.router_config.get('epochs')[0] <= epoch <= \
                       self.model.model.router_config.get('epochs')[1]:
                    # change router
                    pass
                if epoch % 10 == 0 or epoch == self.model.config['epochs'] - 1:
                    x, y_true = zip(*[batch for batch in self.test_set])
                    x = utils.get_y_pred(model.encoder,x)
                    for i, expert in enumerate(model.experts):
                        print(f"Confusion Matrix for Expert {i}")
                        cm = ConfusionMatrix.compute_from_y_pred_y_true(utils.get_y_pred(expert,x), torch.Tensor(y_true))
                        print(cm)
            else:
                self.run_normal_model(epoch)

    def save_results_in_experiment_folder(self, epoch, evaluate_result, path='results.csv'):
        results_csv = os.path.join(self.experiment_path, path)
        df = pd.DataFrame.from_dict(evaluate_result, orient='index').T
        df.insert(0, 'epoch', epoch)
        df.to_csv(results_csv, mode='a', header=not os.path.exists(results_csv), index=False)
