import pandas as pd
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, classification_report
from torch import nn
from transformers import AutoModel, AutoConfig, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


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
 
class MyModel(PreTrainedModel):
    def __init__(self, config):
        super(MyModel, self).__init__(config)
        self.bert = AutoModel.from_pretrained(config.name_or_path)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self._init_weights(self.classifier)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
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
    model_config = AutoConfig.from_pretrained(model_name, num_labels=10)
    model = MyModel(model_config)
    
    train_dataset, validation_dataset = create_datasets(
        file_path=config["dataset"]["file_path"],
        validation_ratio=config["dataset"]["validation_ratio"],
        tokenizer=tokenizer,
        max_length=config["dataset"]["max_length"],
    )
    
    training_args = TrainingArguments(
        bf16=config["training"]["bf16"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
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
        eval_strategy=config["training"]["eval_strategy"],
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
    main(config_path='./configs/trainer_config.yaml')