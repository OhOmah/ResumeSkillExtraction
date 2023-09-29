'''
PURPOSE OF THIS FILE: 

This file will save functions that would be used more than once within the API making the API code much more easy to read and understand. 

This is also the temporary home of the model class. This might be moved to it's own seperate file later. 
'''

from transformers import BertModel

import torch
import transformers

class BERTClass(torch.nn.Module):
    def __init__(self):
        # Defining the layers
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased', return_dict=False)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 22)
    
    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1)
        output_3 = self.l3(output_2)
        return output_3