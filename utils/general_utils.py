import logging
import os.path
import random
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from models.Model import Model

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def run_rl_train_epoch(model: Model, data_loader, scheduler=None) -> float:
    model.train()
    total_loss = 0
    pbar = tqdm(data_loader, desc='Training')
    for batch in pbar:
        model.optimizer.zero_grad()
        loss = model.loss(batch)
        loss.backward()
        model.optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
        pbar.set_postfix({'loss': model.get_loss_repr()})

    total_loss /= len(data_loader)
    logger.debug(f"Train loss: {total_loss}")
    return total_loss


def run_train_epoch(model: Model, data_loader, scheduler=None) -> float:
    model.train()
    total_loss = 0
    pbar = tqdm(data_loader, desc='Training')
    for batch in pbar:
        model.optimizer.zero_grad()
        loss = model.loss(batch)
        loss.backward()
        model.optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
        pbar.set_postfix({'loss': model.get_loss_repr()})

    total_loss /= len(data_loader)
    logger.debug(f"Train loss: {total_loss}")
    return total_loss


def evaluate(model: Model, data_loader) -> dict:
    model.reset_metrics()
    model.eval()
    total_loss = 0
    with torch.no_grad():
        x = 0
        for batch in tqdm(data_loader, desc="Evaluating"):
            loss = model.evaluate(batch)
            total_loss += loss.item()
            x += 1
            if x == 5:
                break
    total_loss /= len(data_loader)
    model_evaluation = model.compute_metrics()
    model_evaluation['loss'] = total_loss
    logger.debug(model_evaluation)
    return model_evaluation


def get_y_pred(model: torch.nn.Module, data_loader) -> (np.ndarray, np.ndarray):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x in data_loader:
            x = x.to(device)
            y_pred.append(model(x).argmax(1).cpu())
    return np.concatenate(y_pred)


def get_y_true_and_y_pred_from_expert(model, data_loader, expert_index) -> (np.ndarray, np.ndarray):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            y_pred.append(model.experts[expert_index](model.encoder(x)).argmax(1).cpu())
            y_true.append(y.cpu())
    return np.concatenate(y_true), np.concatenate(y_pred)


def get_experiment_path(experiment_name) -> str:
    now = datetime.now()
    experiment_name += now.strftime("_%Y-%m-%d_%H-%M-%S")
    experiment_path = os.path.join("experiments", experiment_name)
    return experiment_path
