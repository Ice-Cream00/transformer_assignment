import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
import os
import numpy as np
from data import load_local_csv
from datasets import load_dataset
import time
from model import Seq2SeqTransformer, DecoderLayer, MultiHeadAttention, PositionWiseFFN, ResidualConnection
import logging

# Ensure results dir and logger ready
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
log_file = f"{results_dir}/train_{time.strftime('%Y%m%d_%H%M%S')}.log"
logger = logging.getLogger("transformer_train")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

def train_seq2seq(args, model, get_batch, criterion, optimizer, scheduler, tokenizer):
    """训练seq2seq模型，返回损失记录"""
    os.makedirs("results", exist_ok=True)
    train_losses = []
    val_losses = []
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        # 每轮训练100个批次
        for _ in range(100):
            batch = get_batch("train", args.batch_size, device=args.device)
            src = batch["src"]
            tgt_input = batch["tgt_input"]
            tgt_label = batch["tgt_label"]
            
            optimizer.zero_grad()
            logits = model(src, tgt_input)  # [batch, tgt_seq_len-1, vocab]
            # 计算损失（忽略填充位置）
            loss = criterion(
                logits.reshape(-1, args.tgt_vocab_size),
                tgt_label.reshape(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()
            train_loss += loss.item()
        train_loss /= 100
        train_losses.append(train_loss)
        
        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for _ in range(20):  # 每轮验证20个批次
                batch = get_batch("validation", args.batch_size, device=args.device)
                src = batch["src"]
                tgt_input = batch["tgt_input"]
                tgt_label = batch["tgt_label"]
                
                logits = model(src, tgt_input)
                loss = criterion(
                    logits.reshape(-1, args.tgt_vocab_size),
                    tgt_label.reshape(-1)
                )
                val_loss += loss.item()
        val_loss /= 20
        val_losses.append(val_loss)
        scheduler.step()
        
        print(f"【{args.ablation or '基线模型'}】Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        logger.info(f"[{args.ablation or 'baseline'}] Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    # 保存单模型损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"{args.ablation or 'baseline'} Training Curve (EN-FR Translation)")
    out_path = f"results/{args.ablation or 'baseline'}_loss_en.png"
    plt.savefig(out_path)
    logger.info(f"Saved loss curve: {out_path}")
    plt.close()
    
    return train_losses, val_losses, torch.exp(torch.tensor(val_losses[-1])).item()  # 返回损失和困惑度

def run_seq2seq_experiment(ablation_type=None):
    """运行seq2seq消融实验"""
    args = argparse.Namespace(
        d_model=256,
        num_heads=4,
        num_layers=3,
        max_seq_len=64,
        batch_size=32,
        lr=5e-4,
        epochs=10,  # seq2seq任务训练轮次可减少
        seed=42,
        ablation=ablation_type,
        device="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    # 加载数据
    get_batch, src_vocab_size, tgt_vocab_size, tokenizer, pad_idx, bos_idx, eos_idx = load_local_csv(args.max_seq_len)
    args.src_vocab_size = src_vocab_size
    args.tgt_vocab_size = tgt_vocab_size
    
    # 初始化模型（支持消融实验）
    if args.ablation == "single_head":
        model = Seq2SeqTransformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=args.d_model,
            num_heads=1,  # 单头注意力
            num_layers=args.num_layers,
            pad_idx=pad_idx,
            bos_idx=bos_idx,
            eos_idx=eos_idx
        ).to(args.device)
    elif args.ablation == "no_pos_enc":
        # 移除位置编码
        class DummyPositionalEncoding(nn.Module):
            def __init__(self, *args, **kwargs): super().__init__()
            def forward(self, x): return x
        model = Seq2SeqTransformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            pad_idx=pad_idx,
            bos_idx=bos_idx,
            eos_idx=eos_idx
        ).to(args.device)
        model.encoder.pos_encoding = DummyPositionalEncoding()
        model.decoder.pos_encoding = DummyPositionalEncoding()
    elif args.ablation == "no_cross_attn":
        # Remove encoder-decoder cross attention
        class DummyDecoderLayer(nn.Module):
            def __init__(self, d_model, num_heads, ffn_hidden_dim, dropout=0.1):
                super().__init__()
                self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
                self.ffn = PositionWiseFFN(d_model, ffn_hidden_dim, dropout)
                self.residual1 = ResidualConnection(d_model, dropout)
                self.residual3 = ResidualConnection(d_model, dropout)
            def forward(self, x, enc_out, tgt_mask, src_mask):
                self_attn_output = self.self_attn(x, x, x, tgt_mask)
                x = self.residual1(x, self_attn_output)
                # skip cross attention
                ffn_output = self.ffn(x)
                x = self.residual3(x, ffn_output)
                return x
        model = Seq2SeqTransformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            pad_idx=pad_idx,
            bos_idx=bos_idx,
            eos_idx=eos_idx
        ).to(args.device)
        # 替换 decoder 层为 DummyDecoderLayer
        model.decoder.layers = nn.ModuleList([
            DummyDecoderLayer(args.d_model, args.num_heads, 512, dropout=0.1).to(args.device)
            for _ in range(args.num_layers)
        ])
    else:
        # 基线模型
        model = Seq2SeqTransformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            pad_idx=pad_idx,
            bos_idx=bos_idx,
            eos_idx=eos_idx
        ).to(args.device)
    
    # 优化器和损失函数（忽略填充位置的损失）
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)  # 忽略pad位置的损失
    
    return train_seq2seq(args, model, get_batch, criterion, optimizer, scheduler, tokenizer)

# 运行所有实验
if __name__ == "__main__":
    experiments = {
        "baseline": None,
        "single_head": "single_head",
        "no_pos_enc": "no_pos_enc",
        "no_cross_attn": "no_cross_attn"
    }
    
    results = {}
    for name, ablation in experiments.items():
        print(f"\n===== 开始{name}实验（英法翻译） =====")
        logger.info(f"Start experiment: {name}")
        try:
            train_loss, val_loss, ppl = run_seq2seq_experiment(ablation)
            results[name] = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "perplexity": ppl
            }
        except Exception as e:
            logger.error(f"Experiment {name} failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            print(f"Experiment {name} failed: {e}")
    
    # 打印结果汇总
    logger.info("===== Ablation experiment summary (EN-FR) =====")
    logger.info("| Experiment | Final Val Loss | Perplexity |")
    logger.info("|------------|----------------|------------|")
    for name in experiments.keys():
        loss = results[name]["val_loss"][-1]
        ppl = results[name]["perplexity"]
        logger.info(f"| {name.ljust(8)} | {loss:.4f}       | {ppl:.2f} |")
    
    # Plot validation loss comparison (English)
    plt.figure(figsize=(12, 6))
    for name in experiments.keys():
        if name in results:
            plt.plot(
                results[name]["val_loss"],
                label=f"{name} (Perplexity: {results[name]['perplexity']:.2f})"
            )
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Seq2Seq Ablation Experiment Validation Loss Comparison (EN-FR Translation)")
    plt.legend()
    plt.grid(alpha=0.3)
    comp_path = f"results/seq2seq_ablation_comparison_{time.strftime('%Y%m%d_%H%M%S')}_en.png"
    plt.savefig(comp_path)
    logger.info(f"Saved comparison plot: {comp_path}")
    plt.show()
