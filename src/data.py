def load_iwslt2017(max_seq_len: int = 64):
    """加载IWSLT 2017英法翻译数据集，返回处理后的批次生成函数和词汇表"""
    # 加载数据集（英语→法语）
    dataset = load_dataset("iwslt2017", "iwslt2017-en-fr")
    # 使用预训练分词器（适配英法双语）
    tokenizer = AutoTokenizer.from_pretrained("t5-small", model_max_length=max_seq_len)
    
    # 预处理函数：分词、截断、填充
    def preprocess_function(examples):
        # 源语言：英语（src_text），目标语言：法语（tgt_text）
        src_texts = [ex["en"] for ex in examples["translation"]]
        tgt_texts = [ex["fr"] for ex in examples["translation"]]
        
        # 分词源序列（英语）
        src_encodings = tokenizer(
            src_texts,
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
            return_tensors="pt"
        )
        
        # 分词目标序列（法语）：输入为tgt[:-1]，标签为tgt[1:]（shifted）
        tgt_encodings = tokenizer(
            tgt_texts,
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "src_ids": src_encodings["input_ids"],
            "src_mask": src_encodings["attention_mask"],
            "tgt_input_ids": tgt_encodings["input_ids"][:, :-1],  # 目标输入（不含最后一个token）
            "tgt_labels": tgt_encodings["input_ids"][:, 1:]       # 目标标签（不含第一个token）
        }
    
    # 应用预处理到数据集
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    # 转换为PyTorch格式
    processed_dataset.set_format("torch", columns=["src_ids", "src_mask", "tgt_input_ids", "tgt_labels"])
    
    # 生成批次数据的函数
    def get_batch(split: str, batch_size: int = 32, device: str = "cuda:0"):
        data = processed_dataset[split]
        indices = torch.randperm(len(data))[:batch_size]  # 随机采样batch_size个样本
        batch = {
            "src": data["src_ids"][indices].to(device),
            "tgt_input": data["tgt_input_ids"][indices].to(device),
            "tgt_label": data["tgt_labels"][indices].to(device)
        }
        return batch
    
    # 词汇表大小（源和目标共享分词器，实际可分离）
    src_vocab_size = tokenizer.vocab_size
    tgt_vocab_size = tokenizer.vocab_size
    pad_idx = tokenizer.pad_token_id
    bos_idx = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0
    eos_idx = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1
    
    return get_batch, src_vocab_size, tgt_vocab_size, tokenizer, pad_idx, bos_idx, eos_idx
