# Define the data collator
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling


class ResumableDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def state_dict(self):
        return {"index": self.index}  # Save index position

    def load_state_dict(self, state_dict):
        self.index = state_dict["index"]


class ResumableDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_iteration = 0

    def state_dict(self):
        return {'current_iteration': self.current_iteration}

    def load_state_dict(self, state_dict):
        self.current_iteration = state_dict['current_iteration']
        self.batch_sampler.sampler.set_state(self.current_iteration)


# Create the DataModule
class LLMDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, tokenizer, batch_size=16):
        super().__init__()
        self.train_dataset = ResumableDataset(train_dataset)
        self.val_dataset = ResumableDataset(val_dataset)
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)
        self.batch_size = batch_size

    def train_dataloader(self):
        # return ResumableDataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.data_collator, num_workers=191)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            pin_memory=True,
            num_workers=100 )

    def val_dataloader(self):
        # return ResumableDataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.data_collator,
        #                            num_workers=191)
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            pin_memory=True,
            num_workers=100
        )