import json
import os
from datetime import timedelta
from typing import Dict

import numpy as np
import torch.nn
import torch.utils.data as data
from wandb import AlertLevel

import utils.general_utils as utils
from datasets_and_dataloaders.custom_dataset import CustomDataset
from logger import Logger
from metrics import ConfusionMatrix
from models.MOE import MixtureOfExperts
from models.Model import Model
from utils.singleton_meta import SingletonMeta
from utils.early_stopping import EarlyStopping
import wandb

logger = Logger().logger(__name__)


class Experiment:
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

        self.classes = self.test_set.classes

        self.model = Model(config, train_loader=self.train_loader, test_loader=self.test_loader).to(utils.device)
        # self.model.reset_parameters(dummy_sample.view(1, *dummy_sample.shape).to(utils.device))
        self._init_experiment_folder()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.model.optimizer_exp, T_max=200)  if config.get(
            'scheduler', False) else None
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.model.optimizer, step_size=20, gamma=0.5) if config.get(
        #     'scheduler', False) else None

        print(self.scheduler)
        self.initialized = True
        self.dead_expert_epoch = 0

    def __repr__(self):
        return f"Experiment: {self.config['log']['experiment_name']}"

    def _init_experiment_folder(self):
        self.experiment_path = utils.get_experiment_path(self.config.get('log').get('experiment_name'))
        if not os.path.exists(self.experiment_path):
            os.makedirs(self.experiment_path)
        json.dump(self.config, open(os.path.join(self.experiment_path, "config.json"), 'w'), indent=4)

    def evaluate_and_save_results(self, epoch: int, mode: str, model: torch.nn.Module) -> Dict:
        loader = self.train_loader if mode == 'train' else self.test_loader
        logger.info(f"{mode} evaluation")
        evaluate_result = utils.evaluate(model, loader)
        self.save_results_in_experiment_folder(epoch, evaluate_result=evaluate_result, mode=mode)
        return evaluate_result

    def run_rl_combined_model(self, epoch):
        if epoch == 0:
            train_evaluate_results = self.evaluate_and_save_results(epoch, mode='train', model=self.model)
            validate_evaluate_results = self.evaluate_and_save_results(epoch, mode='test', model=self.model)
            logger.info(f"Train: {train_evaluate_results}")
            logger.info(f"Validate: {validate_evaluate_results}")
        # utils.run_train_epoch(self.model, self.train_loader, self.scheduler)
        if wandb.run:
            wandb.watch(self.model.model)
        self.model.model.train_router(epoch)
        train_evaluate_results = self.evaluate_and_save_results(epoch, mode='train', model=self.model)
        validate_evaluate_results = self.evaluate_and_save_results(epoch, mode='test', model=self.model)
        logger.info(f"Train: {train_evaluate_results}")
        logger.info(f"Validate: {validate_evaluate_results}")
        return train_evaluate_results, validate_evaluate_results


    def log_wandb_logs(self, train_evaluate_results, validate_evaluate_results, model, step):
        wandb_log_dict = {'train': train_evaluate_results, 'validate': validate_evaluate_results}
        if isinstance(model, MixtureOfExperts):
            if 'MOEConfusionMatrix' in validate_evaluate_results.keys():
                wandb_log_dict['validate.MOEConfusionMatrixHeatMap'] = (
                    wandb.plot_table('heatmap', wandb.Table(
                    columns=list(map(str, validate_evaluate_results['MOEConfusionMatrixNormalized'].columns.values)),
                    data=validate_evaluate_results['MOEConfusionMatrixNormalized'].values), {'x': 'columns', 'y': 'index',
                                                                                          'value': 'values'})
                )

                wandb_log_dict['train.MOEConfusionMatrixHeatMap'] =(
                    wandb.plot_table('heatmap', wandb.Table(
                    columns=list(map(str, train_evaluate_results['MOEConfusionMatrixNormalized'].columns.values)),
                    data=train_evaluate_results['MOEConfusionMatrixNormalized'].values), {'x': 'columns', 'y': 'index',
                                                                                          'value': 'values'})
                )
            if 'SuperClassConfusionMatrix' in validate_evaluate_results.keys():
                wandb_log_dict['validate.SuperClassConfusionMatrixHeatMap'] = (
                    wandb.plot_table('heatmap', wandb.Table(
                    columns=list(map(str, validate_evaluate_results['SuperClassConfusionMatrixNormalized'].columns.values)),
                    data=validate_evaluate_results['SuperClassConfusionMatrixNormalized'].values), {'x': 'columns', 'y': 'index',
                                                                                          'value': 'values'})
                )
                wandb_log_dict['train.SuperClassConfusionMatrixHeatMap'] = (
                    wandb.plot_table('heatmap', wandb.Table(
                    columns=list(map(str, train_evaluate_results['SuperClassConfusionMatrixNormalized'].columns.values)),
                    data=train_evaluate_results['SuperClassConfusionMatrixNormalized'].values), {'x': 'columns', 'y': 'index',
                                                                                          'value': 'values'})
                )
            if 'MOEConfusionMatrixNormalized' in validate_evaluate_results.keys() and 'Specialization' in validate_evaluate_results.keys():
                wandb_log_dict['validate.SpecializationHeatMap'] = (
                    wandb.plot_table('heatmap', wandb.Table(
                    columns=list(map(str, validate_evaluate_results['MOEConfusionMatrixNormalized'].columns.values)),
                    data=validate_evaluate_results['Specialization'].values), {'x': 'columns', 'y': 'index',
                                                                                          'value': 'values'})
                )
                wandb_log_dict['train.SpecializationHeatMap'] =(
                    wandb.plot_table('heatmap', wandb.Table(
                    columns=list(map(str, train_evaluate_results['MOEConfusionMatrixNormalized'].columns.values)),
                    data=train_evaluate_results['Specialization'].values), {'x': 'columns', 'y': 'index',
                                                                                          'value': 'values'})
                )
            if False and step % 10 == 0 or step == self.model.config['epochs'] - 1:
                experts_correct_preds_per_class = []
                for i, expert in enumerate(model.experts):
                    logger.debug(f"Confusion Matrix for Expert {i}")
                    labels, preds = utils.get_y_true_and_y_pred_from_expert(model, self.test_loader, i)
                    cm = ConfusionMatrix.compute_from_y_pred_y_true(labels, preds)
                    logger.debug(cm)
                    experts_correct_preds_per_class.append(np.diag(cm) / cm.sum(axis=1))
                if 'MOEConfusionMatrix' in validate_evaluate_results.keys():
                    wandb_log_dict['AccPerExpertHeatMap'] =(
                    wandb.plot_table('heatmap', wandb.Table(
                    columns=list(map(str, train_evaluate_results['MOEConfusionMatrix'].columns.values)),
                    data=np.array(experts_correct_preds_per_class)), {'x': 'columns', 'y': 'index',
                                                                                          'value': 'values'})
                )
        wandb.log(wandb_log_dict)
        wandb.summary.update({'max.validate.Accuracy': max(wandb.summary.get('max.validate.Accuracy', 0), validate_evaluate_results['Accuracy']),
                              'min.validate.Loss': min(wandb.summary.get('min.validate.Loss', float('inf')), validate_evaluate_results['total_loss'])})



    def run_normal_model(self, epoch):
        self.model.model.initial_routing_phase = False
        if self.model.alternate:
            router_phase = epoch % self.model.alternate == 0 and epoch != 0
            self.model.alternate_training_modules(router_phase)
        # self.model.alternate_training_modules(router_phase=True)
        # else:
        #     self.model.alternate_training_modules(False)
        #     self.model.model.initial_routing_phase = True
        # self.model.alternate_training_modules(True)
        if False and epoch == 0:
            train_evaluate_results = self.evaluate_and_save_results(epoch, mode='train', model=self.model)
            validate_evaluate_results = self.evaluate_and_save_results(epoch, mode='test', model=self.model)
            logger.info(f"Train: {train_evaluate_results}")
            logger.info(f"Validate: {validate_evaluate_results}")
            if wandb.run:
                wandb.log({'train': train_evaluate_results, 'validate': validate_evaluate_results})
        utils.run_train_epoch(self.model, self.train_loader, self.scheduler)
        train_evaluate_results = self.evaluate_and_save_results(epoch + 1, mode='train', model=self.model)
        validate_evaluate_results = self.evaluate_and_save_results(epoch + 1, mode='test', model=self.model)
        logger.info(f"Train: {train_evaluate_results}")
        logger.info(f"Validate: {validate_evaluate_results}")
        # if wandb.run:
        #     wandb.watch(self.model.model)
        return train_evaluate_results, validate_evaluate_results

    def run(self):
        model = self.model.model
        # model.router = model.routers[0]
        early_stopping = EarlyStopping(tolerance=10, min_delta=1.)
        best_acc = 0  # float('inf')
        # if isinstance(model, MixtureOfExperts):
        #     for e in model.experts:
        #         e.load_state_dict(torch.load('experiments/baseline_2024-08-20_19-11-25/model.pt'))
        for epoch in range(self.model.config['epochs']):
            while hasattr(model, "current_router_config") and model.current_router_config['epochs'][1] <= epoch:
                model.change_router(model.current_router + 1)
            logger.info(f"Epoch {epoch}")
            if isinstance(model, MixtureOfExperts):
                rl_router = model.unsupervised_router
                if rl_router:
                    train_results, validate_results = self.run_rl_combined_model(epoch)
                else:
                    train_results, validate_results = self.run_normal_model(epoch)
                # routing_entropy = self.model.router_accumulator()
                # if routing_entropy is not None:
                #     np.set_printoptions(threshold=routing_entropy.stats
                #                         .shape[0])
                #     print(f"Routing Entropy: {routing_entropy}")
                # if not self.model.model.router_config.get('epochs')[0] <= epoch <= \
                #        self.model.model.router_config.get('epochs')[1]:
                #     # change router
                #     pass
            else:
                train_results, validate_results = self.run_normal_model(epoch)
            if wandb.run:
                self.log_wandb_logs(train_results, validate_results, model, step=epoch)
            if isinstance(model, MixtureOfExperts):
                if validate_results.get('DeadExperts', 0) == model.num_experts - 1:
                    self.dead_expert_epoch += 1
                else:
                    self.dead_expert_epoch = 0
                if self.dead_expert_epoch > 50 and wandb.run:
                    wandb.alert(
                        title='Dead Experts',
                        text=f'more then 5 epoch with Dead Experts',
                        level=AlertLevel.WARN,
                        wait_duration=timedelta(minutes=5)
                    )
                    break

            if np.isnan(train_results['total_loss']) or np.isnan(validate_results['total_loss']):
                logger.error(f"Loss is nan, breaking")
                break

            # if validate_results['Accuracy'] > best_acc:
            #     best_acc = validate_results['Accuracy']
            #     model_pickle_path = os.path.join(self.experiment_path, 'model.pt')
            #     torch.save(model.state_dict(), model_pickle_path)
            #     print(f"model saved to {model_pickle_path}")
            # early_stopping(train_results['total_loss'], validate_results['total_loss'])
            # if early_stopping.early_stop:
            #     early_stopping(train_results['total_loss'], validate_results['total_loss'])
            #
            #     break

    def save_results_in_experiment_folder(self, epoch: int, evaluate_result: Dict, mode: str):
        result_pickle_path = os.path.join(self.experiment_path, 'results', f'epoch_{epoch}_{mode}.pickle')
        import pickle
        os.makedirs(os.path.dirname(result_pickle_path), exist_ok=True)
        with open(result_pickle_path, 'wb') as f:
            pickle.dump(evaluate_result, f)
