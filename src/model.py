import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
import os
import numpy as np
from datasets import load_dataset

# 1. 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# 2. 多头注意力（支持自注意力和交叉注意力）
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        assert self.d_k * num_heads == d_model, "d_model必须是num_heads的整数倍"
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size = q.size(0)
        # 投影并分多头：[batch, seq_len, d_model] → [batch, num_heads, seq_len, d_k]
        q = self.q_proj(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 缩放点积注意力
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)  # [batch, num_heads, seq_len, seq_len]
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)  # 注意：mask为Bool类型，~mask表示无效位置
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = attn_weights @ v  # [batch, num_heads, seq_len, d_k]
        
        # 拼接多头结果
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_proj(attn_output)

# 3. 前馈网络
class PositionWiseFFN(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# 4. 残差连接 + 层归一化
class ResidualConnection(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer_output: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.dropout(sublayer_output))

# 5. 编码器层
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionWiseFFN(d_model, ffn_hidden_dim, dropout)
        self.residual1 = ResidualConnection(d_model, dropout)
        self.residual2 = ResidualConnection(d_model, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        attn_output = self.self_attn(x, x, x, mask)  # 自注意力
        x = self.residual1(x, attn_output)
        ffn_output = self.ffn(x)
        x = self.residual2(x, ffn_output)
        return x

# 6. 解码器层
class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)  # 掩码自注意力
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)  # 编码器-解码器注意力
        self.ffn = PositionWiseFFN(d_model, ffn_hidden_dim, dropout)
        self.residual1 = ResidualConnection(d_model, dropout)
        self.residual2 = ResidualConnection(d_model, dropout)
        self.residual3 = ResidualConnection(d_model, dropout)

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor, tgt_mask: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # 掩码自注意力（防止关注未来 tokens）
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.residual1(x, self_attn_output)
        # 交叉注意力（关注编码器输出）
        cross_attn_output = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.residual2(x, cross_attn_output)
        # 前馈网络
        ffn_output = self.ffn(x)
        x = self.residual3(x, ffn_output)
        return x

# 7. 编码器
class Encoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int, ffn_hidden_dim: int, max_len: int = 5000, dropout: float = 0.1, pad_idx: int = 0):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, ffn_hidden_dim, dropout) for _ in range(num_layers)])

    def create_pad_mask(self, seq: torch.Tensor) -> torch.Tensor:
        """生成填充掩码（Bool类型：pad位置为False，其他为True）"""
        return (seq != self.pad_idx).unsqueeze(1).unsqueeze(2)  # 关键修改：移除.float()，返回Bool

    def forward(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        src_mask = self.create_pad_mask(src)  # 源序列填充掩码（Bool类型）
        x = self.embedding(src) * math.sqrt(self.d_model)  # 嵌入+缩放
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x, src_mask  # 返回编码器输出和源掩码

# 8. 解码器（核心修改：掩码改为Bool类型）
class Decoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int, ffn_hidden_dim: int, max_len: int = 5000, dropout: float = 0.1, pad_idx: int = 0, bos_idx: int = 1, eos_idx: int = 2):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx  # 句首符号
        self.eos_idx = eos_idx  # 句尾符号
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, ffn_hidden_dim, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)  # 输出投影到目标词汇表

    def create_pad_mask(self, seq: torch.Tensor) -> torch.Tensor:
        """生成填充掩码（Bool类型：pad位置为False，其他为True）"""
        return (seq != self.pad_idx).unsqueeze(1).unsqueeze(2)  # 关键修改：移除.float()，返回Bool

    def create_future_mask(self, seq_len: int) -> torch.Tensor:
        """生成未来掩码（Bool类型：上三角为False，防止关注未来tokens）"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return ~mask.unsqueeze(0).unsqueeze(0)  # 关键修改：移除.float()，返回Bool（~取反使有效位置为True）

    def forward(self, tgt: torch.Tensor, enc_out: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        batch_size, tgt_seq_len = tgt.size()
        # 目标序列掩码：填充掩码 + 未来掩码（均为Bool类型，直接按位与）
        tgt_pad_mask = self.create_pad_mask(tgt)
        tgt_future_mask = self.create_future_mask(tgt_seq_len).to(tgt.device)
        tgt_mask = tgt_pad_mask & tgt_future_mask  # 关键修改：Bool类型直接按位与，解决CUDA冲突
        
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask, src_mask)
        return self.fc_out(x)  # [batch, tgt_seq_len, tgt_vocab_size]

# 9. 完整Seq2Seq Transformer
class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 256, num_heads: int = 4, num_layers: int = 3, ffn_hidden_dim: int = 512, pad_idx: int = 0, bos_idx: int = 1, eos_idx: int = 2):
        super().__init__()
        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_hidden_dim=ffn_hidden_dim,
            pad_idx=pad_idx
        )
        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_hidden_dim=ffn_hidden_dim,
            pad_idx=pad_idx,
            bos_idx=bos_idx,
            eos_idx=eos_idx
        )

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        enc_out, src_mask = self.encoder(src)
        dec_out = self.decoder(tgt, enc_out, src_mask)
        return dec_out
