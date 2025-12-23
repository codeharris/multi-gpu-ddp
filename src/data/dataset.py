# src/data/dataset.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None


# -----------------------------
# Simple hash tokenizer (fast, reproducible, no vocab build)
# -----------------------------
def hash_tokenize(text: str, vocab_size: int, max_len: int) -> torch.Tensor:
    # Basic whitespace tokenization + hashing to [1..vocab_size-1], 0 reserved for PAD
    tokens = text.lower().split()
    ids = []
    for tok in tokens[:max_len]:
        # stable-ish hash across runs: use python hash is salted per process, so use a custom hash
        h = 2166136261
        for c in tok:
            h ^= ord(c)
            h *= 16777619
            h &= 0xFFFFFFFF
        ids.append(1 + (h % (vocab_size - 1)))

    if len(ids) < max_len:
        ids.extend([0] * (max_len - len(ids)))

    return torch.tensor(ids, dtype=torch.long)


# -----------------------------
# Datasets
# -----------------------------
class SyntheticSequenceDataset(Dataset):
    def __init__(self, n_samples: int, seq_len: int, vocab_size: int, num_classes: int, seed: int = 42):
        g = torch.Generator()
        g.manual_seed(seed)
        self.x = torch.randint(low=1, high=vocab_size, size=(n_samples, seq_len), generator=g)
        # Simple deterministic labeling: parity of sum (binary), or modulo for multi-class
        s = self.x.sum(dim=1)
        self.y = (s % num_classes).long()

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class HFTextClassificationDataset(Dataset):
    """
    Wraps a HuggingFace dataset split (with fields: text, label)
    and applies hash tokenization to produce (input_ids, label).
    """
    def __init__(
        self,
        hf_split,
        text_field: str,
        label_field: str,
        vocab_size: int,
        max_len: int,
    ):
        self.ds = hf_split
        self.text_field = text_field
        self.label_field = label_field
        self.vocab_size = vocab_size
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.ds[int(idx)]
        text = item[self.text_field]
        label = int(item[self.label_field])
        x = hash_tokenize(text, self.vocab_size, self.max_len)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


class HFTextClassificationDatasetWithTokenizer(Dataset):
    """
    Wraps a HuggingFace dataset and uses a real tokenizer (for DistilBERT, etc.)
    Returns: (input_ids, attention_mask, label)
    """
    def __init__(
        self,
        hf_split,
        tokenizer,
        text_field: str,
        label_field: str,
        max_len: int,
    ):
        self.ds = hf_split
        self.tokenizer = tokenizer
        self.text_field = text_field
        self.label_field = label_field
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.ds[int(idx)]
        text = item[self.text_field]
        label = int(item[self.label_field])
        
        # Tokenize with transformer tokenizer
        encoded = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        y = torch.tensor(label, dtype=torch.long)
        
        return input_ids, attention_mask, y


# -----------------------------
# Public API
# -----------------------------
def build_dataloaders(cfg: Dict, dist_state) -> Tuple[DataLoader, DataLoader]:
    """
    Builds train/val dataloaders based on cfg["data"]["dataset"].
    Supports:
      - synthetic
      - imdb_hash
      - ag_news
      - ag_news_distilbert (with real tokenizer)
    """
    data_cfg = cfg.get("data", {})
    dataset_name = data_cfg.get("dataset", "synthetic")

    train_bs = int(cfg["training"]["batch_size"])
    max_len = int(cfg["model"]["max_seq_len"])
    vocab_size = int(cfg["model"].get("vocab_size", 30000))
    num_classes = int(cfg["model"]["num_classes"])

    num_workers = int(data_cfg.get("num_workers", 2))
    pin_memory = bool(data_cfg.get("pin_memory", True))
    persistent_workers = num_workers > 0  # Avoid worker teardown
    
    # Check if using DistilBERT
    use_distilbert = (cfg["model"].get("type", "transformer") == "distilbert")

    # -------- build datasets --------
    if dataset_name == "synthetic":
        n_train = int(data_cfg.get("n_train", 20000))
        n_val = int(data_cfg.get("n_val", 5000))
        seq_len = int(data_cfg.get("seq_len", max_len))
        seed = int(cfg.get("seed", 42))

        train_dataset = SyntheticSequenceDataset(
            n_samples=n_train,
            seq_len=seq_len,
            vocab_size=vocab_size,
            num_classes=num_classes,
            seed=seed,
        )
        val_dataset = SyntheticSequenceDataset(
            n_samples=n_val,
            seq_len=seq_len,
            vocab_size=vocab_size,
            num_classes=num_classes,
            seed=seed + 1,
        )

    elif dataset_name in ("imdb_hash", "ag_news", "ag_news_distilbert"):
        if load_dataset is None:
            raise RuntimeError("huggingface 'datasets' is not available. Install it: pip install datasets")

        if dataset_name == "imdb_hash":
            # IMDB: train/test splits. We'll create val from train by small split for quick evaluation.
            raw = load_dataset("imdb")
            text_field = "text"
            label_field = "label"
            train_split = raw["train"]
            test_split = raw["test"]

        else:
            # AG News: train/test (both ag_news and ag_news_distilbert)
            raw = load_dataset("ag_news")
            text_field = "text"
            label_field = "label"
            train_split = raw["train"]
            test_split = raw["test"]

        # Optional: sub-sampling to control runtime (useful for debug)
        train_limit = data_cfg.get("train_limit", None)
        val_limit = data_cfg.get("val_limit", None)
        if train_limit is not None:
            train_split = train_split.select(range(int(train_limit)))
        if val_limit is not None:
            test_split = test_split.select(range(int(val_limit)))

        # Use proper tokenizer for DistilBERT
        if use_distilbert or dataset_name == "ag_news_distilbert":
            from transformers import DistilBertTokenizer
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            
            train_dataset = HFTextClassificationDatasetWithTokenizer(
                hf_split=train_split,
                tokenizer=tokenizer,
                text_field=text_field,
                label_field=label_field,
                max_len=max_len,
            )
            val_dataset = HFTextClassificationDatasetWithTokenizer(
                hf_split=test_split,
                tokenizer=tokenizer,
                text_field=text_field,
                label_field=label_field,
                max_len=max_len,
            )
        else:
            train_dataset = HFTextClassificationDataset(
                hf_split=train_split,
                text_field=text_field,
                label_field=label_field,
                vocab_size=vocab_size,
                max_len=max_len,
            )
            val_dataset = HFTextClassificationDataset(
                hf_split=test_split,
                text_field=text_field,
                label_field=label_field,
                vocab_size=vocab_size,
                max_len=max_len,
            )

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # -------- distributed sampler --------
    if dist_state and getattr(dist_state, "is_distributed", False):
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=dist_state.world_size,
            rank=dist_state.rank,
            shuffle=True,
            drop_last=False,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=dist_state.world_size,
            rank=dist_state.rank,
            shuffle=False,
            drop_last=False,
        )
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_bs,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_bs,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        drop_last=False,
    )

    return train_loader, val_loader
