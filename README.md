# <div align="center"> BERT-Tuner </div>

## A. 介紹

這個 Repo 供了基於 PyTorch 實現的 BERT 多分類任務的最小實現代碼(訓練、推理) & 基於 Transformers 的實現代碼(訓練、推理)，可以根據你的訓練集調整類別數量

## B. 主要功能

### PyTorch 實現

-   **多類分類** : BERT 多分類任務的最小實現
-   **分層學習率衰減(LLRD)** : 可選擇無/簡單/複雜 LLRD，以控制不同層訓練期間學習率
-   **預熱階段** : 支持預熱階段穩定模型早期訓練
-   **無特定框架依賴** : 易調整和擴展的自定義 PyTorch 訓練框架

### Transformers 的實現

-   輕鬆介接 accelerate、deepspeed 等多種訓練算法與框架
-   多種訓練策略

## C. 依賴

依賴皆為常見 library，請自行依賴，以下依賴版本僅為可跑通訓練，其他版本基本上也可以

-   Python 3.10 以上
-   PyTorch
-   Transformers == 4.46.2
-   YAML
-   CUDA（支持 GPU）

## D. 使用方法

### Transformers 的實現(分類任務)

### 訓練

#### 1. 設定 configs/config.yaml

```yaml=
model:
    name: "models/bert-base-chinese"
    num_labels: 10

dataset:
    file_path: "tagged_data_clean.csv"
    validation_ratio: 0.2
    max_length: 128

training:
    output_dir: "./model_for_classification"
    num_train_epochs: 3
    per_device_train_batch_size: 8
    per_device_eval_batch_size: 8
    warmup_ratio: 0.1
    logging_dir: "./logs"
    logging_steps: 10
    save_strategy: "epoch"
    learning_rate: 3e-5
    optim: "adamw_hf"
    eval_steps: 10
    eval_strategy: "steps"
```

#### 2. 數據集格式

數據目前預設是使用 CSV 文件，至少包含兩列(目前 Column 是確定名稱)

| Text | Target                                     |
| ---- | ------------------------------------------ |
| Text | Text # 可參考 tagged_data_clean.csv 資料集 |

#### 3. 啟動訓練

**[可參考該項目來進行混合精度、分布式訓練](https://github.com/FubonDS/Distributed-Parallel-Training)**

```bash=
python BERT_trainer.py --config ./configs/trainer_config.yaml
```

### 推理

**訓練完之後可以通過以下方法進行推理**

```python=
from BERT_inference import BertClassificationInference
model_path = './model_for_classification/checkpoint-216'
    max_length = 128
    num_labels=10
    model = BertClassificationInference(model_path, num_labels=num_labels, max_length=max_length, bf16=True)
    texts = ["都是我們部幫的說的停賣商品了", "都是我們部幫的說的停賣商品了"]
    predictions = model.batch_inference(texts, batch_size=4)
# [4, 4] 類別標籤
```

---

### PyTorch 實現

**在這兩個資料夾下分別為分類任務(`./BERT_Training_single_cls`)、多標籤分類任務(`BERT_Training_multi_cls`)代碼**

-   **各自資料夾下有說明文檔**
