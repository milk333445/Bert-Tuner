import pandas as pd
import yaml
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, classification_report
from torch import nn
from transformers import AutoConfig, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding, PreTrainedModel

from model import BertClassificationModel


def load_config(file_path):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


class MyDataset(Dataset):
    def __init__(self, tokenizer, data, max_length=512):
        self.dataset = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __getitem__(self, index):
        data = self.dataset.iloc[index]
        source = data['Text']
        target = data['Target']
        
        # tokenizer
        encoded = self.tokenizer(source, padding='max_length', max_length=self.max_length, truncation=True, return_tensors='pt')        
        encoded = {key: val.squeeze(0) for key, val in encoded.items()}
        encoded['labels'] = torch.tensor(target, dtype=torch.long)
        
        return encoded
    
    def __len__(self):
        return len(self.dataset)

    
def create_datasets(file_path, validation_ratio, tokenizer, max_length, random_state=42):
    pd_data = pd.read_csv(file_path)
    pd_validation_data = pd_data.sample(frac=validation_ratio, random_state=random_state).reset_index(drop=True)
    pd_traindata = pd_data[~pd_data.index.isin(pd_validation_data.index)].reset_index(drop=True)
    train_dataset = MyDataset(tokenizer, pd_traindata, max_length)
    validation_dataset = MyDataset(tokenizer, pd_validation_data, max_length)
    return train_dataset, validation_dataset


def compute_metrics(pred):
    logits, labels = pred
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)
    predictions = torch.argmax(logits, dim=-1)
    return {
        'accuracy': accuracy_score(labels.numpy(), predictions.numpy()),
    }


def main(config_path='./configs/trainer_config.yaml'):
    config = load_config(config_path)
    
    model_name = config["model"]["name"]
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_config = AutoConfig.from_pretrained(model_name, num_labels=config["model"]["num_labels"])
    model = BertClassificationModel(model_config)
    
    train_dataset, validation_dataset = create_datasets(
        file_path=config["dataset"]["file_path"],
        validation_ratio=config["dataset"]["validation_ratio"],
        tokenizer=tokenizer,
        max_length=config["dataset"]["max_length"],
    )
    
    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        num_train_epochs=config["training"]["num_train_epochs"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        warmup_ratio=config["training"]["warmup_ratio"],
        logging_dir=config["training"]["logging_dir"],
        logging_steps=config["training"]["logging_steps"],
        save_strategy=config["training"]["save_strategy"],
        learning_rate=float(config["training"]["learning_rate"]),
        optim=config["training"]["optim"],
        eval_steps=config["training"]["eval_steps"],
        evaluation_strategy=config["training"]["eval_strategy"],
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer) 
    )
    trainer.train()
    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training BERT model') 
    parser.add_argument('--config', type=str, default='./configs/trainer_config.yaml', help='config file path')
    args = parser.parse_args()
    main(config_path=args.config)   