import torch
from torch import nn
import os
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

from transformers import AdamW, get_linear_schedule_with_warmup


def to_device(dict_tensors, device):
    result_tensors = {}
    for k, v in dict_tensors.items():
        result_tensors[k] = v.to(device)
    return result_tensors

def validate(model, validation_loader, criteria, device, threshold=0.5):
    model.eval()
    total_loss = 0
    total_correct = 0
    all_targets = []
    all_predictions = []
    
    for inputs, targets in validation_loader:
        inputs = to_device(inputs, device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = criteria(outputs, targets)
        
        total_loss += float(loss.item())
        
        probablities = F.softmax(outputs, dim=1)
        predictions = (probablities > threshold).float() 
        all_predictions.extend(predictions.cpu().numpy())   
        all_targets.extend(targets.cpu().numpy())
        
    all_predictions = torch.tensor(all_predictions)
    all_targets = torch.tensor(all_targets)
    
    # 樣本準確率(要求所有類別都正確)
    sample_accuracy = accuracy_score(all_targets, all_predictions)
        
    # 標籤準確率(只要有一個類別正確即可)
    correct_labels = (all_targets == all_predictions).sum()
    total_labels = all_targets.numel()
    label_accuracy = correct_labels / total_labels
    
    precision = precision_score(all_targets, all_predictions, average="samples", zero_division=0)
    recall = recall_score(all_targets, all_predictions, average="samples", zero_division=0)
    f1 = f1_score(all_targets, all_predictions, average="samples", zero_division=0)
    
    classification_report_str = classification_report(all_targets, all_predictions, zero_division=0)
    
    print("Classification Report:")
    print(classification_report_str)
    
    return sample_accuracy, label_accuracy, total_loss / len(validation_loader.dataset), classification_report_str
    
        
        

def train(model, train_loader, validation_loader, criteria, optimizer, device, epochs, log_per_step, model_dir, scheduler=None):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    total_loss = 0
    step = 0
    best_accuracy = 0
    best_classification_report = None
    model_dir = Path(model_dir)
    os.makedirs(model_dir, exist_ok=True)
    for epoch in range(epochs):
        model.train()
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = to_device(inputs, device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criteria(outputs, targets)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
            total_loss += float(loss.item())
            step += 1
            if step % log_per_step == 0:
                print(f"Epoch {epoch+1}/{epochs}, Step: {i}/{len(train_loader)}, Total Loss: {total_loss:.4f}")
                total_loss = 0
            del inputs, targets, outputs
        sample_accuracy, label_accuracy, validation_loss, classification_report = validate(model, validation_loader, criteria, device)
        print(f"Epoch {epoch+1}/{epochs}, Validation Sample Accuracy: {sample_accuracy:.4f}, Validation Label Accuracy: {label_accuracy:.4f}, Validation Loss: {validation_loss:.4f}")
        torch.save(model, model_dir / f"model_{epoch}.pth")
        if sample_accuracy > best_accuracy:
            torch.save(model, model_dir / f"model_best.pth")
            best_accuracy = sample_accuracy
            best_classification_report = classification_report
        torch.cuda.empty_cache()
        

"""
Classification Report:
              precision    recall  f1-score   support

           0       1.00      0.93      0.96        14
           1       1.00      1.00      1.00         3
           2       1.00      1.00      1.00         6
           3       1.00      1.00      1.00         1
           4       0.00      0.00      0.00         1
           5       1.00      1.00      1.00         9
           6       1.00      1.00      1.00        10
           7       1.00      0.83      0.91         6
           8       1.00      0.86      0.92         7
           9       0.79      1.00      0.88        15

    accuracy                           0.94        72
   macro avg       0.88      0.86      0.87        72
weighted avg       0.94      0.94      0.94        72
"""