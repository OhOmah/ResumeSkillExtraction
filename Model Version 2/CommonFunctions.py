"""
PURPOSE OF THIS FILE: 
This file exists to contain all the declarations of custom functions and classes 
defined in this project.
"""
from torch.utils.data import Dataset
import torch
import transformers


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


class BERTClass(torch.nn.Module, len):
    def __init__(self):
        # Defining the layers
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained(
            "bert-base-uncased", return_dict=False
        )
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(len, 22)

    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1)
        output_3 = self.l3(output_2)
        return output_3


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)
