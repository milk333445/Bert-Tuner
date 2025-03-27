import os
import torch
from transformers import AutoTokenizer, AutoConfig
from tqdm import tqdm
from model import BertClassificationModel


class BertClassificationInference:
    def __init__(self, model_path, num_labels=10, max_length=128, bf16=False, fp16=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if bf16 and self.device.type == "cuda":
            torch_dtype = torch.bfloat16
        elif fp16 and self.device.type == "cuda":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
            
        self.model_config = AutoConfig.from_pretrained(model_path, num_labels=num_labels)
        self.model = BertClassificationModel.from_pretrained(model_path, config=self.model_config).to(self.device, dtype=torch_dtype).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_length = max_length
        
    def batch_inference(self, texts, batch_size=4):
        if not isinstance(texts, list):
            texts = [texts]
        
        all_predictions = []
        all_probabilities = []
        
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i + batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            inputs = self.to_device(inputs)
            
            with torch.no_grad():
                outputs = self.model(inputs['input_ids'], inputs['attention_mask'])
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(probabilities, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_probabilities.extend(probabilities.detach().to(torch.float).cpu().numpy().tolist())
            
            del inputs, outputs
        torch.cuda.empty_cache()    
        return all_predictions

    def to_device(self, input_data):
        ret = {}
        for k, v in input_data.items():
            v = v.to(self.model.device)
            if torch.is_floating_point(v):
                v = v.to(self.model.dtype)
            ret[k] = v
        return ret   


if __name__ == "__main__":
    model_path = './model_for_classification/checkpoint-216'
    max_length = 128
    num_labels=10
    model = BertClassificationInference(model_path, num_labels=num_labels, max_length=max_length, bf16=True)
    texts = ["都是我們部幫的說的停賣商品了", "都是我們部幫的說的停賣商品了", "都是我們部幫的說的停賣商品了", "都是我們部幫的說的停賣商品了", "都是我們部幫的說的停賣商品了", "都是我們部幫的說的停賣商品了"]
    predictions = model.batch_inference(texts, batch_size=4)
    print(predictions)

    
        
    