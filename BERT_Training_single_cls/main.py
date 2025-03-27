import yaml
import torch
from torch import nn
from torch.optim import Adam


from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup


from CustomModel import MyModel
from CustomDataset import load_data, create_dataloaders
from trainer import train


def main(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    device = torch.device("cuda" if torch.cuda.is_available() and config['device'] == 'cuda' else 'cpu')
    print(f'Using device: {device}')
    train_data, validation_data = load_data(config['data_file'], config['validation_ratio'])
    tokenizer = AutoTokenizer.from_pretrained(config['model_path'])
    train_loader, validation_loader = create_dataloaders(train_data, validation_data, tokenizer, config['batch_size'], config['text_max_length'], num_workers=config['num_workers'], pin_memory=config['pin_memory'])
    model = MyModel(config['model_path'], config['num_labels']).to(device)
    num_training_steps = len(train_loader) * config['epochs']
    num_warmup_steps = int(num_training_steps * float(config['warmup_ratio']))
    criteria = nn.CrossEntropyLoss()
    if config['LLDR_complex']:
        print('Using LLRD_complex')
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
        optimizer = Adam(optimizer_grouped_parameters)         
    elif config['LLRD_simple'] == True and config['LLDR_complex'] == False:
        print('Using LLDR_simple')
        bert_params = model.bert.parameters()
        predictor_params = model.predictor.parameters()
        optimizer = torch.optim.AdamW([
            {'params': bert_params, 'lr': float(config['bert_learning_rate'])}, 
            {'params': predictor_params, 'lr': float(config['predictor_learning_rate'])}  
        ])
    else:
        print('Using AdamW')
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['learning_rate']))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    train(model, train_loader, validation_loader, criteria, optimizer, device, config['epochs'], config['log_per_step'], config['output_dir'], scheduler)

if __name__ == "__main__":
    main('./configs/config.yaml')