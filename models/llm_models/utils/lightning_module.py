import pytorch_lightning as pl
import torch


class LightningModule(pl.LightningModule):
    def __init__(self, model, tokenizer, optimizers, schedulers=None, lr = 0.001, additional_loss_function = None,  **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.tokenizer = tokenizer
        self._optimizers = optimizers
        self._schedulers = schedulers
        self._lr = lr
        self._additional_loss_function = additional_loss_function

    def forward(self, input_ids, attention_mask=None, labels=None, output_hidden_states=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                          output_hidden_states=output_hidden_states)

    def training_step(self, batch, batch_idx):
        batch.update({"output_hidden_states": True})
        # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # outputs = self.model(input_ids, attention_mask=attention_mask)
        outputs = self(**batch)

        if self._additional_loss_function is not None:
            additional_loss = self._additional_loss_function(self.model, outputs, batch)
            self.log("Additional Loss", additional_loss, prog_bar=True, sync_dist=True)


        self.log("CE loss", outputs.loss, prog_bar=True, sync_dist=True)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        if self._optimizers is not None:
            return self._optimizers
        optimizer =  torch.optim.AdamW(self.model.parameters(), lr=self._lr)
        return optimizer
