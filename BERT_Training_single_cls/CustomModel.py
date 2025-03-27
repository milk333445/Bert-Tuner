import torch
from torch import nn
from transformers import AutoModel, AutoConfig


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
            nn.Softmax(dim=1)
        )
    
    def forward(self, src):
        outputs = self.bert(**src).last_hidden_state[:, 0, :]
        return self.predictor(outputs)