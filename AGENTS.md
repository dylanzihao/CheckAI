# CheckAI Agents Guide

## 项目概览

训练 BERT（chinese-roberta-wwm-ext）用于中文 AI 文本检测（二分类：Human vs AI-Generated）。

### 三个阶段

| 阶段 | 执行环境 | 说明 |
|------|----------|------|
| 数据处理 | 本地 | 清洗、划分、增强 |
| 训练 | Kaggle | 通过 run.ipynb 从 GitHub 克隆代码并执行 |
| 推理 | 本地 | INT8 量化 + 推理，针对 Intel GPU/NPU 优化 |

## 目录结构规范

```
CheckAI/
├── data/
│   ├── raw/
│   │   └── C-ReD/               # 原始数据，只读不修改
│   ├── processed/               # 清洗划分后的数据（JSONL 格式）
│   │   ├── train.jsonl
│   │   ├── val.jsonl
│   │   └── test.jsonl
│   └── augmented/               # 数据增强后的训练数据（JSONL 格式）
│       └── train_augmented.jsonl
├── src/
│   ├── data/                    # 数据处理脚本
│   │   ├── clean.py             # 数据清洗
│   │   ├── split.py             # 数据集划分
│   │   └── augment.py           # nlpcda 数据增强
│   ├── train/                   # 训练相关代码
│   │   ├── config.py            # 训练配置（路径解析 + TrainingArguments）
│   │   └── trainer.py           # 训练主逻辑（HuggingFace Trainer API）
│   ├── inference/               # 推理相关代码
│   │   ├── quantize.py          # INT8 量化脚本
│   │   └── predict.py           # 推理脚本
│   └── run.ipynb                # Kaggle 训练入口 Notebook
├── models/                      # 模型保存（gitignored）
│   ├── base/                    # 微调后的 FP32 模型
│   └── quantized/               # INT8 量化模型
├── requirements/                # 依赖文件
│   ├── requirements-local.txt   # 本地依赖（数据处理 + 推理）
│   └── requirements-kaggle.txt  # Kaggle 依赖（训练）
├── AGENTS.md
└── README.md
```

## 数据处理规范

### 数据源

- **C-Red** (`data/raw/C-ReD/benchmark data/`)
- 5 个类别：`composition`, `film_review`, `news`, `paper`, `question_answer`
- 9 种 AI 模型：`claude-3.5-haiku`, `deepseek-r1`, `deepseek-v3`, `doubao-1.5-pro`, `gemini-2.5-flash`, `gpt-3.5-turbo`, `gpt-4o`, `qwen-2.5`, `qwen-3`
- label: 1 = human, 0 = AI-generated

### 数据清洗

- 去除空文本或极短文本（长度 < 10 字符）
- 去除重复文本（基于 text 去重）
- 统一列名和格式，human 数据补齐 `prompt` 列为空
- 统一 `label` 字段（int 类型）
- 去除 HTML 标签、多余空白字符
- 确保所有 CSV 列结构一致

### 数据集划分

- **比例**: 8:1:1（训练:验证:测试）
- **训练集目标**: ~30K 条
- **采样策略**: 按类别均匀采样
  - 5 个类别各采 1/5，约 6000 条/类
  - 每类内 human:AI = 1:1（约 3000 human + 3000 AI）
- **分层划分**: 按类别逐层划分，保证各 split 中类别分布和 human/AI 比例一致

### 数据增强（nlpcda）

- 仅对**训练集**进行增强
- 增强方法：全部使用
  - **同义词替换** (SimbertBased): 生成语义相近的句子
  - **随机插入** (RandomInsert): 随机位置插入词语
  - **随机删除** (RandomDelete): 随机删除部分词语
  - **随机交换** (RandomSwap): 交换句子中词语位置
  - **TF-IDF 替换** (TfIdfBased): 基于 TF-IDF 相似词替换
- 增强倍率：每个原始样本生成 2-3 个增强样本
- 注意事项：
  - 增强后的训练集需保持 1:1 平衡
  - 增强样本的 label 与原始样本一致
  - 验证集和测试集不做增强

## 训练规范

### 模型

- **预训练模型**: `hfl/chinese-roberta-wwm-ext`
- **本地路径**: `D:\Dylan\Model\chinese-roberta-wwm-ext`
- **模型类**: `AutoModelForSequenceClassification`（HuggingFace 内置）
- **任务**: 二分类（AI-generated text detection）
- **损失函数**: CrossEntropyLoss（模型内置）

### 框架

- PyTorch + Transformers + Datasets + HuggingFace Trainer API
- 多 GPU 训练：Trainer 内置 DDP，使用 torchrun 启动（2×T4）

### 超参配置

```
max_seq_length: 512               # BERT 最大长度，覆盖 ~63% 数据无需截断
batch_size: 8                     # per GPU，2×T4 共 16; T4 16GB 下 8 为安全值
gradient_accumulation_steps: 2    # 有效 batch_size = 8 × 2 GPU × 2 = 32
learning_rate: 2e-5
num_epochs: 3
optimizer: AdamW
scheduler: linear warmup + linear decay
warmup_ratio: 0.1
fp16: true                        # T4 支持混合精度加速
```

### run.ipynb（Kaggle 入口）

该 Notebook 实现从 GitHub 克隆源码并完整执行训练流程：

1. **从 GitHub 克隆**当前仓库到 Kaggle 环境
2. **安装依赖**：`pip install -r /kaggle/working/CheckAI/requirements/requirements-kaggle.txt`
3. **（可选）预加载模型**：将 Kaggle Input 中的模型转为 safetensors 缓存到 `/kaggle/working/`，加速训练子进程加载
4. **启动训练**：使用 `torch.distributed.run` 启动 `trainer.py`（内部使用 HuggingFace Trainer API）
5. **Trainer 自动处理**：分布式初始化、混合精度、梯度累积、日志、评估、checkpoint 保存

## 推理规范

### 量化

- 使用 `optimum-intel` + `openvino` + `nncf` 实现 INT8 量化
- 目标硬件：Intel GPU / NPU
- 量化策略：
  - 使用 NNCF 进行 INT8 PTQ（Post-Training Quantization）
  - 需要校准数据集：从验证集中抽取 200-500 条，覆盖各类别和 Human/AI 两种标签，不重复使用训练集
  - 导出为 OpenVINO IR 格式
  - 针对 Intel GPU (OpenCL) 和 NPU 进行推理优化
- 输出：量化后的 OpenVINO IR 模型（`.xml` + `.bin`），以及校准后的精度对比报告

### 推理

- 支持 FP32（原始模型）和 INT8（量化模型）两种模式
- 输入：单条文本 或 CSV 批量文件
- 输出：label（0/1）和置信度概率
- 支持 Intel GPU / NPU 设备选择
- 性能对比：输出 FP32 vs INT8 的推理速度和精度对比

## 代码规范

- **优先使用标准库**：能用 HuggingFace / PyTorch 内置 API 的不要自己手写（如 `Trainer` > 手动训练循环、`AutoModelForSequenceClassification` > 自定义 `PreTrainedModel`、`datasets.Dataset` > 自定义 `torch.utils.data.Dataset`）。只有标准库无法满足需求时才自行实现
- 使用清晰的函数和类命名，建议添加类型注解
- 关键配置使用 config 或 yaml 管理，避免硬编码
- 数据处理脚本应为可复现的模块化设计
- 日志使用标准的 logging 模块，输出关键中间结果和数据统计
- 所有本地路径统一使用 `pathlib.Path` 管理

## 依赖管理

### 本地依赖（数据处理 + 推理）

- pandas, numpy
- nlpcda
- torch
- transformers
- optimum-intel, openvino, nncf
- pathlib

### Kaggle 依赖（训练）

- torch
- transformers
- datasets
- pandas, numpy
- scikit-learn
- tqdm

## 程序交付使用

所有脚本从项目根目录（`CheckAI/`）执行。

### 数据处理

```bash
# 1. 数据清洗：读取 raw/C-ReD 所有 CSV，清洗后按类别保存为 JSONL
python src/data/clean.py

# 2. 数据集划分：读取清洗后的数据，分层采样划分 8:1:1
#    输出 data/processed/train.jsonl / val.jsonl / test.jsonl
python src/data/split.py

# 3. 数据增强：对训练集进行 nlpcda 增强
#    输出 data/augmented/train_augmented.jsonl
python src/data/augment.py
```

### 训练

```bash
# 本地调试（单机 CPU/GPU）
python src/train/trainer.py

# Kaggle 提交：将 run.ipynb 上传至 Kaggle，连接 GPU 加速器后运行全部单元格
# Notebook 会自动从 GitHub 克隆源码、安装依赖、加载数据、分布式训练并保存模型
```

### 推理

```bash
# 1. INT8 量化：加载微调后的 FP32 模型，用 NNCF 量化并导出 OpenVINO IR
#    需要校准文件（从验证集抽取的 200-500 条 JSONL）
#    输出 models/quantized/ 下的 .xml + .bin
python src/inference/quantize.py \
    --model-path models/base \
    --calibration-file data/processed/val.jsonl \
    --calibration-samples 300 \
    --output-dir models/quantized

# 2. 单条文本预测
python src/inference/predict.py --model-path models/quantized --text "待检测文本"

# 3. 批量文件预测（JSONL 格式）
python src/inference/predict.py --model-path models/quantized --input-file data/test.jsonl --output-file results.jsonl

# 4. 对比 FP32 与 INT8 性能
python src/inference/predict.py --model-path models/base --text "待检测文本"
python src/inference/predict.py --model-path models/quantized --text "待检测文本"
```
