import pandas as pd
import torch
from transformers import AutoTokenizer

def load_local_csv(
    max_seq_len: int = 64,
    csv_path: str = '././train_8000_en2fr.csv',
    val_ratio: float = 0.1,
    test_ratio: float = 0.1
):
    """
    加载本地英法翻译CSV，自动划分训练、验证、测试集，返回批次生成函数和词汇表
    """
    df = pd.read_csv(csv_path)
    src_texts = df['src'].tolist()
    tgt_texts = df['tgt'].tolist()
    n = len(src_texts)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    n_train = n - n_val - n_test

    train_idx = list(range(0, n_train))
    val_idx = list(range(n_train, n_train + n_val))
    test_idx = list(range(n_train + n_val, n))

    splits = {
        'train': train_idx,
        'validation': val_idx,
        'test': test_idx
    }

    try:
        tokenizer = AutoTokenizer.from_pretrained("t5-small", model_max_length=max_seq_len)
        # 额外测试tokenizer是否能正常工作（防止部分环境下from_pretrained不报错但实际无法encode）
        _ = tokenizer([src_texts[0]], truncation=True, max_length=max_seq_len, padding="max_length", return_tensors="pt")
    except Exception as e:
        print(f"[WARN] HuggingFace tokenizer unavailable, using local word tokenizer. Reason: {e}")
        all_texts = src_texts + tgt_texts
        tokenizer = LocalWordTokenizer(all_texts, max_length=max_seq_len)

    def preprocess(indices):
        src_batch = [src_texts[i] for i in indices]
        tgt_batch = [tgt_texts[i] for i in indices]
        src_encodings = tokenizer(src_batch, truncation=True, max_length=max_seq_len, padding="max_length", return_tensors="pt")
        tgt_encodings = tokenizer(tgt_batch, truncation=True, max_length=max_seq_len, padding="max_length", return_tensors="pt")
        return {
            "src": src_encodings["input_ids"],
            "tgt_input": tgt_encodings["input_ids"][:, :-1],
            "tgt_label": tgt_encodings["input_ids"][:, 1:]
        }

    def get_batch(split: str, batch_size: int = 32, device: str = "cuda:0"):
        idx_pool = splits.get(split, train_idx)
        indices = torch.randperm(len(idx_pool))[:batch_size]
        batch_indices = [idx_pool[i] for i in indices]
        batch = preprocess(batch_indices)
        batch = {k: v.to(device) for k, v in batch.items()}
        return batch

    src_vocab_size = tokenizer.vocab_size
    tgt_vocab_size = tokenizer.vocab_size
    pad_idx = tokenizer.pad_token_id
    bos_idx = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0
    eos_idx = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1

    return get_batch, src_vocab_size, tgt_vocab_size, tokenizer, pad_idx, bos_idx, eos_idx