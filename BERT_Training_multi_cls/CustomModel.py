import torch
from torch import nn
from transformers import AutoModel, AutoConfig
import random


class MyModel(nn.Module):
    def __init__(self, model_path, num_labels, load_pretrained=True):
        super(MyModel, self).__init__()
        if load_pretrained:
            self.bert = AutoModel.from_pretrained(model_path)
        else:
            config = AutoConfig.from_pretrained(model_path)
            self.bert = AutoModel.from_config(config)
        self.num_bert_layers = self.bert.config.num_hidden_layers
        self.predictor = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, num_labels),
            # nn.Sigmoid() # 因為用 bcewithlogitsloss 所以不需要 sigmoid
        )
    
    def forward(self, src):
        outputs = self.bert(**src).last_hidden_state[:, 0, :]
        return self.predictor(outputs)
    
if __name__ == "__main__":
    model = MyModel('./models/bert-base-chinese', 5)
    x = torch.randint(0, 100, (2, 128)) # batch size = 2, sequence length = 128
    attention_mask = torch.randint(0, 2, (2, 128)) # batch size = 2, sequence length = 128
    outputs = model({'input_ids': x, 'attention_mask': attention_mask})
    print(outputs)