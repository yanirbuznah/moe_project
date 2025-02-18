from typing import Union, Optional, Callable, Any

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch.optim import Optimizer
from transformers import DataCollatorForLanguageModeling, DynamicCache
import torch

from .modeling_deepseek import DeepseekMoE, AddAuxiliaryLoss


class DualOptimizerDeepseekLightningModule(pl.LightningModule):
    def __init__(self, model, tokenizer, lr=0.001):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = lr

    def forward(self, input_ids, attention_mask=None, labels=None):
        # input_ids = input_ids.unsqueeze(0)
        # attention_mask = attention_mask.unsqueeze(0)
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    #
    # def training_step(self, batch, batch_idx):
    #     outputs = self(**batch)
    #     self.compute_additional_loss(outputs)
    #     self.log('train_loss', loss, prog_bar=True, sync_dist=True)
    #     return loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        outputs = self(**batch)  # Assume y1 is for loss1 and y2 is for loss2

        if optimizer_idx == 0:
            loss = outputs.loss
            self.log("CE loss", loss, prog_bar=True, sync_dist=True)

        if optimizer_idx == 1:
            loss = self.compute_additional_loss(outputs, batch)
            self.log("Gain Loss", loss, prog_bar=True)

        return outputs.loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def compute_additional_loss(self, outputs, batch):
        # Implement your additional loss computation here
        # return torch.tensor(0.0)  # Placeholder
        gain_losses = []
        batch.update({"output_hidden_states": True})
        for i in range(1, len(outputs.hidden_states) - 1):
            batch_new = batch.copy()
            batch_new.update(
                {"inputs_embeds": outputs.hidden_states[i], "decoder_layer_idx": i, "past_key_values": DynamicCache(),
                 'use_cache': False,
                 'random_routing': 'random_weighted'})
            del batch_new["input_ids"]
            with torch.no_grad():
                # self.model.eval()
                outputs_new = self. model(**batch_new)
                # print(torch.equal(outputs_new.hidden_states[1], outputs.hidden_states[i+1]))
            gain_loss = outputs.loss - outputs_new.loss
            if gain_loss > 0:
                AddAuxiliaryLoss(self.model.model.layers[i].mlp.gate.logits, gain_loss)
                gain_losses.append(gain_loss)
        return torch.stack(gain_losses).mean() if gain_losses else torch.tensor(0.0)


    def configure_optimizers(self):
        optimizer_params = [{'params': layer.mlp.gate.parameters()} for layer in self.model.model.layers if
                            isinstance(layer.mlp, DeepseekMoE)]
        optimizer1 = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        optimizer2 = torch.optim.AdamW(optimizer_params, lr=self.learning_rate)
        return [optimizer1, optimizer2]


