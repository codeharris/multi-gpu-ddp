from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import torch


class AmazonPolarityHFDataset(Dataset):
    """
    Amazon Reviews Polarity with Hugging Face tokenizer.

    - Loads 'amazon_polarity' via HuggingFace datasets.
    - Concatenates title + content, tokenizes with pretrained tokenizer.
    - Pads/truncates to max_seq_len, returns attention masks.
    """

    def __init__(self, split: str, tokenizer_name: str, max_seq_len: int):
        ds = load_dataset("amazon_polarity")[split]
        titles = ds["title"]
        contents = ds["content"]
        self.texts = [f"{t} {c}" if t else c for t, c in zip(titles, contents)]
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
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        return item, label
