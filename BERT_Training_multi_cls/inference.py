import os
import torch
import pandas as pd
import argparse
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
import time
import copy
import torch.nn.functional as F

class InferenceModel:
    def __init__(self, directory_path, max_length=128):
        model_file = next((f for f in os.listdir(directory_path) if f.endswith('.pth')), None)
        if model_file is None:
            raise FileNotFoundError("No .pth file found in the directory.")
        model_path = os.path.join(directory_path, model_file)
        self.model = torch.load(model_path)
        self.model.eval()
        # fp16
        self.model.half()
        self.tokenizer = AutoTokenizer.from_pretrained(directory_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_length = max_length

    def predict(self, texts, threshold=0.5):
        if not isinstance(texts, list):
            texts = [texts]
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        with torch.no_grad():  
            outputs = self.model(inputs) 
            outputs = F.sigmoid(outputs).cpu()
            predictions = (outputs > threshold).float().cpu().numpy()
            outputs = outputs.detach().numpy()

        del inputs
        return outputs, predictions
    
def main(args):
    model = InferenceModel(args.model_path, args.max_length)
    pd_data = pd.read_csv(args.data_path)
    texts = pd_data['Text'].tolist()
    preds = []
    
    start_time = time.time()
    for i in tqdm(range(0, len(texts), args.batch_size)):
        batch_texts = texts[i:i+args.batch_size]
        output, pred = model.predict(batch_texts, args.threshold)
        preds.extend(pred)
    print(f'time spend: {time.time()-start_time}')
    print(f'time spend per sample: {(time.time()-start_time)/len(texts)}')
    
    pd_data['pred'] = preds
    pd_data.to_csv(args.output_path, index=False, encoding='utf-8-sig')
    print(f"Predictions saved to {args.output_path}")

if __name__ == "__main__":
    """
    single inference
    texts = ["都是我們部幫的說的停賣商品了", "都是我們部幫的說的停賣商品了"]
    outputs, preds = model.predict(texts)
    """
    parser = argparse.ArgumentParser(description="Run predictions on csv file.")
    parser.add_argument("--model_path", default='best_bert_model', type=str,
                        help="Path to the pre-trained model.")
    parser.add_argument("--data_path", default='example_dataset.csv', type=str,
                        help="Path to the CSV file containing the texts to predict.")
    parser.add_argument("--output_path", type=str, default='example_dataset_preds.csv',
                        help="Path where the predictions will be saved.")
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for prediction.')
    
    parser.add_argument('--max_length', type=int, default=256, help='Max length of the input text.')
    
    parser.add_argument('--batch_size', type=int, default=5, help='batch_inference for inferencer.')
    args = parser.parse_args()
    main(args)
    
        
    