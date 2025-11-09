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
    
    # 保存单模型损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="训练损失")
    plt.plot(val_losses, label="验证损失")
    plt.xlabel("轮次")
    plt.ylabel("损失值")
    plt.legend()
    plt.title(f"{args.ablation or '基线模型'}训练曲线（英法翻译）")
    plt.savefig(f"results/{args.ablation or 'baseline'}_loss.png")
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
    get_batch, src_vocab_size, tgt_vocab_size, tokenizer, pad_idx, bos_idx, eos_idx = load_iwslt2017(args.max_seq_len)
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
        # 移除编码器-解码器注意力
        class DummyDecoderLayer(DecoderLayer):
            def forward(self, x, enc_out, tgt_mask, src_mask):
                self_attn_output = self.self_attn(x, x, x, tgt_mask)
                x = self.residual1(x, self_attn_output)
                # 跳过交叉注意力
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
        model.decoder.layers = nn.ModuleList([
            DummyDecoderLayer(args.d_model, args.num_heads, 512)
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
        train_loss, val_loss, ppl = run_seq2seq_experiment(ablation)
        results[name] = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "perplexity": ppl
        }
    
    # 打印结果汇总
    print("\n===== 消融实验结果汇总（英法翻译） =====")
    print(f"| 实验配置 | 最终验证损失 | 困惑度 |")
    print(f"|----------|--------------|--------|")
    for name in experiments.keys():
        loss = results[name]["val_loss"][-1]
        ppl = results[name]["perplexity"]
        print(f"| {name.ljust(8)} | {loss:.4f}       | {ppl:.2f} |")
    
    # 绘制验证损失对比曲线
    plt.figure(figsize=(12, 6))
    for name in experiments.keys():
        plt.plot(
            results[name]["val_loss"],
            label=f"{name} (困惑度: {results[name]['perplexity']:.2f})"
        )
    plt.xlabel("训练轮次")
    plt.ylabel("验证损失")
    plt.title("seq2seq消融实验验证损失对比（英法翻译）")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("results/seq2seq_ablation_comparison.png")
    plt.show()
