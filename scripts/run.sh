#!/bin/bash

# 运行基线模型
python src/train.py \
  --d_model 256 \
  --num_heads 4 \
  --num_layers 3 \
  --seq_len 64 \
  --batch_size 32 \
  --lr 5e-4 \
  --epochs 10 \
  --seed 42

# 运行消融实验（可选，取消注释即可）
# python src/train.py \
#   --ablation single_head \
#   --d_model 256 \
#   --num_layers 4 \
#   --seq_len 64 \
#   --batch_size 32 \
#   --lr 5e-4 \
#   --epochs 10 \
#   --seed 42

# python src/train.py \
#   --ablation no_pos_enc \
#   --d_model 256 \
#   --num_heads 4 \
#   --num_layers 3 \
#   --seq_len 64 \
#   --batch_size 32 \
#   --lr 5e-4 \
#   --epochs 10 \
#   --seed 42

# python src/train.py \
#   --ablation no_cross_attn \
#   --d_model 256 \
#   --num_heads 4 \
#   --num_layers 3 \
#   --seq_len 64 \
#   --batch_size 32 \
#   --lr 5e-4 \
#   --epochs 10 \
#   --seed 42
