# <div align="center"> BERT多標籤分類訓練最小實現 </div>

## A. 介紹
這個Repo供了基於PyTorch實現的BERT多標籤分類任務的最小實現代碼，可以根據你的訓練集調整類別數量

## B. 主要功能
- **多標籤分類** : BERT多標籤分類任務的最小實現
- **分層學習率衰減(LLRD)** : 可選擇無/簡單/複雜LLRD，以控制不同層訓練期間學習率
- **預熱階段** : 支持預熱階段穩定模型早期訓練
- **無特定框架依賴** : 易調整和擴展的自定義PyTorch訓練框架

## C. 依賴
依賴皆為常見library，請自行依賴，以下依賴版本僅為可跑通訓練，其他版本基本上也可以
- Python 3.10以上
- PyTorch == 2.3.1+cu121
- Transformers == 4.46.2
- YAML
- CUDA（支持GPU）

## D. 使用方法
### 1. 設定configs/config.yaml
```yaml=
batch_size: 8
text_max_length: 128
num_labels: 10
epochs: 30
learning_rate: 3e-5
warmup_ratio: 0.1
validation_ratio: 0.1
device: cuda
log_per_step: 50
model_path: models/bert-base-chinese # 要微調模型，僅支持BERT系列
data_file: example_dataset.csv # 訓練資料集
output_dir: output/bert_checkpoints # 微調後模型權重存放路徑
LLRD_simple: False # 使用簡單LLRD，BERT層與最後分類層使用不同學習率
bert_learning_rate: 5e-5 # BERT層學習率
predictor_learning_rate: 1e-4 # 分類層學習率
LLDR_complex: False # 使用複雜LLRD，自頂上下，學習率逐層衰減decay_factor
base_learning_rate: 2e-5
decay_factor: 0.95
```

### 2. 數據集格式
數據目前預設是使用CSV文件，至少包含兩列(目前Column是確定名稱)

| Text | Target |
| ---- | ------ |
| Text | [1, 2] # ，可以參考example_dataset.csv   |

### 3. 啟動訓練
運行main.py腳本，會根據config.yaml的設定開始訓練，並訓練完後保存到指定資料夾
```python=
python main.py
```
