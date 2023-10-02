import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset



def read_and_prepare_data():
    '''
    PURPOSE OF THIS FUNCTION: 
    Intakes and prepares the data to be used for training and testing for the Llama model. 
    This is here to save space in the notebook for ease of readability. A deep dive into test 
    and training data analysis and manipulation can be found in 
    "Model Version 2/Data manipulation V2.ipynb"
    '''
    test_df = pd.read_csv("../Data/TestingData.csv")
    train_df = pd.read_csv("../Data/Training_Data.csv")
    label_df = pd.read_csv("../Data/label_df.csv")

    train_df['Label'] = train_df['Label'].apply(lambda s: [float(x.strip(' []')) for x in s.split(',')])
    test_df['Label'] = test_df['Label'].apply(lambda s: [float(x.strip(' []')) for x in s.split(',')])


    return test_df, train_df, label_df

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.reported_jobs = dataframe.Reported_Jobs
        self.targets = dataframe.Label
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
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

class Loss(nn.Module):
    def __init(self):
        super(Loss, self).__init__()

    def forward(self, inputs, targets):
        loss = -1 * (targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs))
        return loss.mean()
