from datasets import load_dataset
from torch.utils.data import Dataset
import torch


class IMDBHashDataset(Dataset):
    """
    IMDB dataset with a simple hash-based tokenizer.

    - Loads the 'imdb' dataset via HuggingFace datasets.
    - Tokenizes text by splitting on whitespace.
    - Maps each token to an integer: hash(token) % vocab_size.
    - Pads/truncates to max_seq_len.

    This avoids building a full vocabulary and keeps things fast and simple
    for HPC performance experiments.
    """

    def __init__(self, split: str, vocab_size: int, max_seq_len: int):
        """
        Args:
            split: 'train' or 'test'
            vocab_size: size of the hash-based vocabulary
            max_seq_len: maximum sequence length for padding/truncation
        """
        dataset = load_dataset("imdb")[split]
        self.texts = dataset["text"]
        self.labels = dataset["label"]
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.texts)

    def _text_to_ids(self, text: str) -> torch.Tensor:
        # Simple whitespace tokenization
        tokens = text.split()

        # Convert tokens to ids using hash trick
        ids = [hash(tok) % self.vocab_size for tok in tokens]

        # Truncate
        ids = ids[: self.max_seq_len]

        # Pad
        if len(ids) < self.max_seq_len:
            ids = ids + [0] * (self.max_seq_len - len(ids))

        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]  # 0 or 1

        input_ids = self._text_to_ids(text)
        return input_ids, label
