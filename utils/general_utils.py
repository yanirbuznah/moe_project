import random

import numpy as np
import torch
from tqdm import tqdm

from models.Model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def run_train_epoch(model: Model, data_loader, scheduler=None):
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader, desc="Training epoch"):
        model.optimizer.zero_grad()
        loss = model.loss(batch)
        loss.backward()
        model.optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()

    total_loss /= len(data_loader)
    print(f"Train loss: {total_loss}")
    return total_loss


def evaluate(model: Model, data_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            model_evaluation = model.evaluate(batch)
            total_loss += model_evaluation['loss'].item()
    total_loss /= len(data_loader)
    model_evaluation['loss'] = total_loss
    print(model_evaluation)
    return model_evaluation
