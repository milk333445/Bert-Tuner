import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import AutoModel, AutoConfig, AutoTokenizer, PreTrainedModel
import os
from pathlib import Path
from tqdm import tqdm

from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report

import yaml

from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup

# ddp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
# dist.init_process_group(backend='nccl')

# accelerate
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

import math

def validate(model, validation_loader, accelerator, criteria):
    model.eval()
    total_loss = torch.tensor(0.0, device=accelerator.device)
    all_targets = []
    all_predictions = []
    acc_num = 0
    with torch.no_grad():
        for inputs, targets in validation_loader:
            outputs = model(inputs)
            loss = criteria(outputs, targets)
            total_loss += float(loss.item())
            _, predicted = torch.max(outputs, 1)

            predicted, targets = accelerator.gather_for_metrics((predicted, targets))
            acc_num += (predicted == targets).float().sum()
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    total_samples = len(validation_loader.dataset)

    # ddp
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM) # format: tensor 單卡會報錯

    classification_report_str = classification_report(all_targets, all_predictions, zero_division=0)
    accelerator.print(classification_report_str)
    return acc_num / total_samples, total_loss / total_samples, classification_report_str

def train(model, train_loader, validation_loader, accelerator, criteria, optimizer, epochs, log_per_step, model_dir, scheduler=None, resume=None):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    total_loss = 0
    step = 0
    best_accuracy = 0
    best_classification_report = None
    model_dir = Path(model_dir)
    os.makedirs(model_dir, exist_ok=True)
    
    resume_step = 0
    resume_epoch = 0
    
    if resume is not None:
        accelerator.load_state(resume)
        steps_per_epoch = math.ceil(len(train_loader) / accelerator.gradient_accumulation_steps)
        resume_step = step = int(resume.split("step_")[-1])
        resume_epoch = resume_step // steps_per_epoch
        resume_step -= resume_epoch * steps_per_epoch
        accelerator.print(f'resume from checkpoint -> {resume}')
    
    for epoch in range(resume_epoch, epochs):
        model.train()
        if resume and epoch == resume_epoch and resume_step != 0:
            activate_dataloader = accelerator.skip_first_batches(train_loader, resume_step * accelerator.gradient_accumulation_steps)
        else:
            activate_dataloader = train_loader
        
        for i, (inputs, targets) in enumerate(activate_dataloader):
            with accelerator.accumulate(model):
                outputs = model(inputs)
                loss = criteria(outputs, targets)
                accelerator.backward(loss)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

                total_loss += float(loss.item())
                if accelerator.sync_gradients:
                    step += 1
                    if step % log_per_step == 0:
                        loss = accelerator.reduce(loss, "mean")
                        accelerator.print(f"Epoch {epoch+1}/{epochs}, Step: {i}/{len(train_loader)}, Total Loss: {loss.item()}")
                        accelerator.log({"loss": loss.item()}, step)
                        total_loss = 0

                    if step % 40 == 0:
                        accelerator.print(f'save checkpoint -> step {step}')
                        accelerator.save_state(accelerator.project_dir + f"/step_{step}")
                        accelerator.unwrap_model(model).save_pretrained(
                            save_directory=accelerator.project_dir + f"/step_{step}/model",
                            is_main_process=accelerator.is_main_process,
                            state_dict=accelerator.get_state_dict(model),
                            save_func=accelerator.save
                        )
                del inputs, targets, outputs
        accuracy, validation_loss, classification_report = validate(model, validation_loader, accelerator, criteria)
        accelerator.print(f"Epoch {epoch+1}/{epochs}, Validation Accuracy: {accuracy:.4f}, Validation Loss: {validation_loss:.4f}")
        accelerator.log({"validation_accuracy": accuracy}, step)
         
        if accuracy > best_accuracy:
            # torch.save(model, model_dir / f"model_best.pth")
            best_accuracy = accuracy
            best_classification_report = classification_report
        torch.cuda.empty_cache()

# class MyModel(nn.Module):
#     def __init__(self, model_path, num_labels):
#         super(MyModel, self).__init__()
#         self.bert = AutoModel.from_pretrained(model_path)
#         self.num_bert_layers = self.bert.config.num_hidden_layers
#         self.predictor = nn.Sequential(
#             nn.Linear(768, 256),
#             nn.ReLU(),
#             nn.Linear(256, num_labels),
#             # nn.Softmax(dim=1) 因為後面用crossentropyloss，所以不需要softmax
#         )

#     def forward(self, src):
#         outputs = self.bert(**src).last_hidden_state[:, 0, :]
#         return self.predictor(outputs)

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

    def forward(self, src):
        outputs = self.bert(**src).last_hidden_state[:, 0, :]
        outputs = self.dropout(outputs)
        return self.classifier(outputs)


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

def load_data(file_path, validation_ratio, random_state=42):
    # random_state 確保ddp不同進程取樣相同
    pd_data = pd.read_csv(file_path)
    pd_validation_data = pd_data.sample(frac=validation_ratio, random_state=random_state).reset_index(drop=True)
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, collate_fn=collate, shuffle=False)
    return train_loader, validation_loader

def main(config_path):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=2,
        kwargs_handlers=[kwargs],
        log_with="tensorboard",
        project_dir='./ckpt'
        )
    
    accelerator.init_trackers("runs")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    device = torch.device("cuda" if torch.cuda.is_available() and config['device'] == 'cuda' else 'cpu')
    accelerator.print(f'Using device: {device}')
    train_data, validation_data = load_data(config['data_file'], config['validation_ratio'])
    tokenizer = AutoTokenizer.from_pretrained(config['model_path'])
    train_loader, validation_loader = create_dataloaders(train_data, validation_data, tokenizer, config['batch_size'], config['text_max_length'], num_workers=config['num_workers'], pin_memory=config['pin_memory'])
    # model = MyModel(config['model_path'], config['num_labels'])
    model_config = AutoConfig.from_pretrained(config['model_path'], num_labels=config['num_labels'])
    model = MyModel(model_config)    

    num_training_steps = len(train_loader) * config['epochs']
    num_warmup_steps = int(num_training_steps * float(config['warmup_ratio']))
    criteria = nn.CrossEntropyLoss()
    if config['LLDR_complex']:
        accelerator.print('Using LLRD_complex')
        optimizer_grouped_parameters = []
        layerwise_learning_rate = float(config['base_learning_rate'])
        for layer_index in range(model.num_bert_layers):
            layer_parameters = {
                'params': [p for n, p in model.bert.named_parameters() if f'.layer.{layer_index}.' in n],
                'lr': layerwise_learning_rate
            }
            optimizer_grouped_parameters.append(layer_parameters)
            layerwise_learning_rate *= float(config['decay_factor'])
        custom_layer_params = list(model.predictor.parameters())
        optimizer_grouped_parameters.append({
            'params': custom_layer_params,
            'lr': layerwise_learning_rate 
        })
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters)         
    elif config['LLRD_simple'] == True and config['LLDR_complex'] == False:
        accelerator.print('Using LLDR_simple')
        bert_params = model.bert.parameters()
        predictor_params = model.predictor.parameters()
        optimizer = torch.optim.AdamW([
            {'params': bert_params, 'lr': float(config['bert_learning_rate'])}, 
            {'params': predictor_params, 'lr': float(config['predictor_learning_rate'])}  
        ])
    else:
        accelerator.print('Using AdamW')
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['learning_rate']))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # accelerate
    model, optimizer, train_loader, validation_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, validation_loader, scheduler)
    
    train(model, train_loader, validation_loader, accelerator, criteria, optimizer, config['epochs'], config['log_per_step'], config['output_dir'], scheduler, resume=None)

    accelerator.end_training()

if __name__ == "__main__":
    main('./configs/config.yaml')