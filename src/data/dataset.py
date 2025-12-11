import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


class SyntheticSequenceDataset(Dataset):
    """
    Simple synthetic sequence classification dataset.

    - Each sample is a sequence of token IDs of length seq_len.
    - Vocabulary size = vocab_size.
    - Label rule:
        If sum(token_ids) > threshold -> label 1
        else -> label 0
    """

    def __init__(self, size: int, seq_len: int, vocab_size: int, threshold_factor: float = 0.5):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        # Threshold for synthetic labels
        self.max_sum = (vocab_size - 1) * seq_len
        self.threshold = self.max_sum * threshold_factor

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Random integer sequence [0, vocab_size)
        x = torch.randint(
            low=0,
            high=self.vocab_size,
            size=(self.seq_len,),
            dtype=torch.long,
        )

        seq_sum = x.sum().item()
        y = 1 if seq_sum > self.threshold else 0

        return x, y


def build_dataloaders(cfg: dict, dist_state):
    """
    Build train and validation DataLoaders.
    Uses DistributedSampler if DDP is enabled.
    """

    seq_len = cfg["model"].get("max_seq_len", 64)
    vocab_size = cfg["model"].get("vocab_size", 1000)
    batch_size = cfg["training"]["batch_size"]

    train_dataset = SyntheticSequenceDataset(
        size=2000,
        seq_len=seq_len,
        vocab_size=vocab_size,
        threshold_factor=0.5,
    )

    val_dataset = SyntheticSequenceDataset(
        size=400,
        seq_len=seq_len,
        vocab_size=vocab_size,
        threshold_factor=0.5,
    )

    # Distributed samplers for DDP
    if dist_state.use_ddp:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=dist_state.world_size,
            rank=dist_state.rank,
            shuffle=True,
        )
        val_sampler = None
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
    )

    return train_loader, val_loader
