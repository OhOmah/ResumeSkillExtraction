"""
PURPOSE OF THIS FILE: 
This file exists to contain all the declarations of custom functions and classes 
defined in this project.
"""
from torch.utils.data import Dataset
import torch


class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.reported_jobs = dataframe.Reported_Jobs
        self.targets = self.data.Label
        self.max_len = max_len

    def __len__(self):
        return len(self.reported_jobs)

    def __getitem__(self, index):
        reported_job = str(self.reported_jobs[index])
        reported_job = " ".join(reported_job.split())

        inputs = self.tokenizer.encode_plus(
            reported_job,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.targets[index], dtype=torch.float),
        }
