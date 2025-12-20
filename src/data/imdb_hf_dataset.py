from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import torch


class IMDBHFDataset(Dataset):
    """
    IMDB dataset with Hugging Face tokenizer for pretrained models.

    - Loads 'imdb' via HuggingFace datasets.
    - Uses a pretrained tokenizer (e.g., distilbert-base-uncased).
    - Pads/truncates to max_seq_len with attention masks.
    """

    def __init__(self, split: str, tokenizer_name: str, max_seq_len: int):
        ds = load_dataset("imdb")[split]
        self.texts = ds["text"]
        self.labels = ds["label"]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = int(self.labels[idx])
        enc = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # squeeze batch dimension
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        return item, label
