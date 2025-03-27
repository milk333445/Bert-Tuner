from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("ckiplab/bert-base-chinese")
model = AutoModel.from_pretrained("ckiplab/bert-base-chinese")
model.save_pretrained("./models/bert-base-chinese")
tokenizer.save_pretrained("./models/bert-base-chinese")
