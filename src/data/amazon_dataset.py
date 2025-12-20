from datasets import load_dataset
from torch.utils.data import Dataset
import torch


class AmazonPolarityHashDataset(Dataset):
    """
    Amazon Reviews Polarity dataset with a simple hash-based tokenizer.

    - Loads the 'amazon_polarity' dataset via HuggingFace datasets.
    - Tokenizes text by splitting on whitespace (title + content).
    - Maps each token to an integer: hash(token) % vocab_size.
    - Pads/truncates to max_seq_len.

    Train split size: ~3.6M rows, Test split: ~400k rows.
    """

    def __init__(self, split: str, vocab_size: int, max_seq_len: int):
        """
        Args:
            split: 'train' or 'test'
            vocab_size: size of the hash-based vocabulary
            max_seq_len: maximum sequence length for padding/truncation
        """
        dataset = load_dataset("amazon_polarity")[split]
        # Concatenate title and content for richer signal
        titles = dataset["title"]
        contents = dataset["content"]
        self.texts = [f"{t} {c}" if t else c for t, c in zip(titles, contents)]
        self.labels = dataset["label"]  # 0 or 1
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.texts)

    def _text_to_ids(self, text: str) -> torch.Tensor:
        tokens = text.split()
        ids = [hash(tok) % self.vocab_size for tok in tokens]
        ids = ids[: self.max_seq_len]
        if len(ids) < self.max_seq_len:
            ids = ids + [0] * (self.max_seq_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = int(self.labels[idx])
        input_ids = self._text_to_ids(text)
        return input_ids, label
