# Transformer 实现与实验

本项目手工实现了完整的 Transformer 模型（编码器-解码器架构），并在 IWSLT 2017 数据集上进行英法机器翻译任务。

## 项目结构

```
transformer_assignment-main/
├── src/                    # 源码目录（模型、数据、训练）
│   ├── model.py
│   ├── data.py
│   └── train.py
├── scripts/                # 运行脚本
│   └── run.sh
├── results/                # 结果目录（训练后自动生成）
│   ├── *_loss_en.png       # 各实验损失曲线
│   └── seq2seq_ablation_comparison_*.png  # 对比图
│   └── train_xxx.log  # 训练日志
├── train_8000_en2fr.csv    # 本地英法翻译数据集（8000对）
├── requirements.txt        # 依赖包
└── README.md               # 本文件
```

## 数据集

本项目使用 **IWSLT 2017** 数据集进行机器翻译任务训练。

- **数据集类型**: 序列到序列（Sequence-to-Sequence）机器翻译
- **数据集名称**: `iwslt2017`，配置 `iwslt2017-en-fr`
- **语言对**: 英语→法语（English → French）
- **Hugging Face 链接**: [iwslt2017](https://huggingface.co/datasets/iwslt2017)
- **分词器**: 使用 `t5-small` 预训练分词器进行子词分词
- **特点**: 
  - 适合训练编码器-解码器 Transformer 模型
  - 使用预训练分词器，支持子词级别的处理
  - 包含真实的翻译数据，适合评估模型性能

数据集会自动从 Hugging Face 下载，无需手动准备。

## 硬件要求

- **CPU**: 任意现代 CPU 即可
- **GPU**: 可选，支持 CUDA 的 GPU 可加速训练（推荐）
- **内存**: 至少 4GB RAM
- **存储**: 至少 1GB 可用空间（用于下载数据集和模型）

## 环境配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

**注意**: 本项目使用 `datasets==3.6.0` 版本以确保兼容性。如果遇到数据集加载问题，请确保使用正确的版本：

```bash
pip install --upgrade --force-reinstall --no-cache-dir torch==2.4.0 datasets==3.6.0 fsspec==2024.6.1 matplotlib==3.7.1 numpy==1.26.4 transformers==4.36.2
```

### 2. 验证安装

```bash
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import datasets; print(f'Datasets版本: {datasets.__version__}')"
```

## 运行方式

### 快速开始

使用提供的脚本运行所有实验（包括基线模型和消融实验）：

```bash
bash scripts/run.sh
```

脚本会自动运行以下实验：
- 基线模型（baseline）
- 单头注意力实验（single_head）
- 移除位置编码实验（no_pos_enc）
- 移除交叉注意力实验（no_cross_attn）

### 完整命令（可复现实验）

基线模型训练（包含随机种子）：

```bash
python src/train.py \
  --d_model 256 \
  --num_heads 4 \
  --num_layers 3 \
  --seq_len 64 \
  --batch_size 32 \
  --lr 5e-4 \
  --epochs 10 \
  --seed 42
```

**参数说明：**
- `--d_model`: 模型维度（默认：256）
- `--num_heads`: 注意力头数（默认：4）
- `--num_layers`: 编码器/解码器层数（默认：3）
- `--seq_len`: 序列最大长度（默认：64）
- `--batch_size`: 批次大小（默认：32）
- `--lr`: 学习率（默认：5e-4）
- `--epochs`: 训练轮数（默认：10）
- `--seed`: 随机种子（默认：42）

### 消融实验

代码会自动运行所有消融实验。如果需要单独运行某个实验，可以修改 `train.py` 中的 `experiments` 字典。

#### 1. 单头注意力实验

```bash
python src/train.py \
  --ablation single_head \
  --d_model 256 \
  --num_layers 3 \
  --seq_len 64 \
  --batch_size 32 \
  --lr 5e-4 \
  --epochs 10 \
  --seed 42
```

#### 2. 移除位置编码实验

```bash
python src/train.py \
  --ablation no_pos_enc \
  --d_model 256 \
  --num_heads 4 \
  --num_layers 3 \
  --seq_len 64 \
  --batch_size 32 \
  --lr 5e-4 \
  --epochs 10 \
  --seed 42
```

#### 3. 移除交叉注意力实验

```bash
python src/train.py \
  --ablation no_cross_attn \
  --d_model 256 \
  --num_heads 4 \
  --num_layers 3 \
  --seq_len 64 \
  --batch_size 32 \
  --lr 5e-4 \
  --epochs 10 \
  --seed 42
```

## 模型架构

### 核心组件

1. **位置编码 (Positional Encoding)**: 正弦/余弦位置编码
2. **多头自注意力 (Multi-Head Self-Attention)**: 缩放点积注意力机制
3. **位置无关前馈网络 (Position-wise FFN)**: 两层全连接网络
4. **残差连接与层归一化 (Residual Connection & Layer Normalization)**
5. **编码器层 (Encoder Layer)**: 自注意力 + FFN
6. **解码器层 (Decoder Layer)**: 掩码自注意力 + 交叉注意力 + FFN

### 默认超参数

- `d_model`: 256（模型维度）
- `num_heads`: 4（注意力头数）
- `num_layers`: 3（编码器/解码器层数）
- `ffn_hidden_dim`: 512（前馈网络隐藏层维度）
- `max_seq_len`: 64（序列最大长度）
- `batch_size`: 32（批次大小）
- `lr`: 5e-4（学习率）
- `epochs`: 10（训练轮数）

### 数据处理

- **分词器**: 使用 `t5-small` 预训练分词器
- **词汇表大小**: 约 32,000（由分词器决定）

#### 数据加载

使用预训练分词器进行子词级别的分词，支持更高效的词汇表管理和更好的泛化能力。

## 实验设置

- **随机种子**: 42（确保可复现性）
- **设备**: 自动检测 CUDA，否则使用 CPU
- **任务**: 英法机器翻译（English → French）

## 依赖包版本

为确保兼容性，本项目使用以下特定版本：

- `torch==2.4.0`
- `datasets==3.6.0`
- `fsspec==2024.6.1`
- `matplotlib==3.7.1`
- `numpy==1.26.4`
- `transformers==4.36.2`

