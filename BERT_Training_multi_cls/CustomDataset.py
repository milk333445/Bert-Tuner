import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from CustomModel import MyModel
import torch.nn as nn



def pad_target(target_list, num_classes):
    target_tensor = torch.zeros(num_classes)
    for idx in target_list:
        target_tensor[int(idx)] = 1
    return target_tensor


class MyDataset(Dataset):
    def __init__(self, data, mode="train", num_classes=5):
        self.dataset = data
        self.mode = mode
        self.num_classes = num_classes
        
    def __getitem__(self, index):
        data = self.dataset.iloc[index]
        source = data['Text']
        target = pad_target(data['Target'], self.num_classes)
        # target = torch.tensor(data['Target'], dtype=torch.float) 
        return source, target
    
    def __len__(self):
        return len(self.dataset)


def load_data(file_path, validation_ratio):
    pd_data = pd.read_csv(file_path)
    import ast
    pd_data['Target'] = pd_data['Target'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    pd_validation_data = pd_data.sample(frac=validation_ratio).reset_index(drop=True)
    pd_traindata = pd_data[~pd_data.index.isin(pd_validation_data.index)].reset_index(drop=True)
    return pd_traindata, pd_validation_data

def collate_fn(batch, tokenizer, text_max_length):
    text, target = zip(*batch)
    src = tokenizer(list(text), padding='max_length', max_length=text_max_length, truncation=True, return_tensors='pt')
    return src, torch.stack(target)

def create_dataloaders(train_data, validation_data, tokenizer, batch_size, text_max_length, num_workers=1, pin_memory=False, num_classes=5):
    from functools import partial
    collate = partial(collate_fn, tokenizer=tokenizer, text_max_length=text_max_length)

    train_dataset = MyDataset(train_data, num_classes=num_classes)
    validation_dataset = MyDataset(validation_data, num_classes=num_classes)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=num_workers, pin_memory=pin_memory)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, validation_loader


if __name__ == "__main__":
    model = MyModel('./models/bert-base-chinese', 5)
    train_data, validation_data = load_data("combined_obstacle_detection_data.csv", 0.2)
    tokenizer = AutoTokenizer.from_pretrained("./models/bert-base-chinese")
    train_loader, validation_loader = create_dataloaders(train_data, validation_data, tokenizer, 2, 128, num_classes=5)
    
    criterion = nn.BCEWithLogitsLoss()
    for batch in train_loader:
        src, target = batch
        outputs = model(src)
        print(outputs)
        loss = criterion(outputs, target)
        print(loss) # 0.7030