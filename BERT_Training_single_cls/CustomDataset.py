import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class MyDataset(Dataset):
    def __init__(self, data, mode="train"):
        self.dataset = data
        self.mode = mode
        
    def __getitem__(self, index):
        data = self.dataset.iloc[index]
        source = data['Text']
        target = data['Target']
        return source, target
    
    def __len__(self):
        return len(self.dataset)

def load_data(file_path, validation_ratio):
    pd_data = pd.read_csv(file_path)
    pd_validation_data = pd_data.sample(frac=validation_ratio).reset_index(drop=True)
    pd_traindata = pd_data[~pd_data.index.isin(pd_validation_data.index)].reset_index(drop=True)
    return pd_traindata, pd_validation_data

def collate_fn(batch, tokenizer, text_max_length):
        text, target = zip(*batch)
        src = tokenizer(list(text), padding='max_length', max_length=text_max_length, truncation=True, return_tensors='pt')
        return src, torch.LongTensor(target)

def create_dataloaders(train_data, validation_data, tokenizer, batch_size, text_max_length, num_workers=1, pin_memory=False):

    from functools import partial
    collate = partial(collate_fn, tokenizer=tokenizer, text_max_length=text_max_length)

    train_dataset = MyDataset(train_data)
    validation_dataset = MyDataset(validation_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=num_workers, pin_memory=pin_memory)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, validation_loader