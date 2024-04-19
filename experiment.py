import json
import os
from typing import Dict

import torch.nn
import torch.utils.data as data
import utils.general_utils as utils
from datasets_and_dataloaders.custom_dataset import CustomDataset
from logger import Logger
from metrics import ConfusionMatrix
from models.MOE import MixtureOfExperts
from models.Model import Model
from utils.singleton_meta import SingletonMeta
import wandb

logger = Logger().logger(__name__)


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

    def evaluate_and_save_results(self, epoch: int, mode: str, model: torch.nn.Module):
        loader = self.train_loader if mode == 'train' else self.test_loader
        logger.info(f"{mode} evaluation)")
        evaluate_result = utils.evaluate(model, loader)
        self.save_results_in_experiment_folder(epoch, evaluate_result=evaluate_result, mode=mode)
        return evaluate_result

    def run_rl_combined_model(self, epoch):
        utils.run_train_epoch(self.model, self.train_loader)
        wandb.watch(self.model.model)
        self.model.model.train_router(epoch)
        train_evaluate_results = self.evaluate_and_save_results(epoch, mode='train', model=self.model)
        validate_evaluate_results = self.evaluate_and_save_results(epoch, mode='test', model=self.model)
        logger.info(f"Train: {train_evaluate_results}")
        logger.info(f"Validate: {validate_evaluate_results}")
        if wandb.run:
            wandb.log({'train': train_evaluate_results, 'validate': validate_evaluate_results})
    def run_normal_model(self, epoch):
        utils.run_train_epoch(self.model, self.train_loader)

        train_evaluate_results = self.evaluate_and_save_results(epoch, mode='train', model=self.model)
        validate_evaluate_results = self.evaluate_and_save_results(epoch, mode='test', model=self.model)
        logger.info(f"Train: {train_evaluate_results}")
        logger.info(f"Validate: {validate_evaluate_results}")
        if wandb.run:
            wandb.log({'train': train_evaluate_results, 'validate': validate_evaluate_results})

    def run(self):
        model = self.model.model
        for epoch in range(self.model.config['epochs']):
            logger.info(f"Epoch {epoch}")
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
                    for i, expert in enumerate(model.experts):
                        logger.debug(f"Confusion Matrix for Expert {i}")
                        cm = ConfusionMatrix.compute_from_y_pred_y_true(
                            *utils.get_y_true_and_y_pred_from_expert(model, self.test_loader, i))
                        logger.debug(cm)
            else:
                self.run_normal_model(epoch)

    def save_results_in_experiment_folder(self, epoch: int, evaluate_result: Dict, mode: str):
        result_pickle_path = os.path.join(self.experiment_path, 'results', f'epoch_{epoch}_{mode}.pickle')
        import pickle
        os.makedirs(os.path.dirname(result_pickle_path), exist_ok=True)
        with open(result_pickle_path, 'wb') as f:
            pickle.dump(evaluate_result, f)
