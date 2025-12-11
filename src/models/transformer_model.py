import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


class TransformerSequenceClassifier(nn.Module):
    """
    Transformer encoder for sequence classification.

    - Input: token IDs (batch, seq_len)
    - Output: logits over classes (batch, num_classes)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        dim_feedforward: int,
        num_classes: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (batch, seq, dim)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids):
        """
        input_ids: (batch, seq_len), dtype long
        """
        x = self.embedding(input_ids)      # (batch, seq_len, d_model)
        x = self.pos_encoder(x)
        x = self.encoder(x)                # (batch, seq_len, d_model)

        # Pool over the sequence: simple mean
        x = x.mean(dim=1)                  # (batch, d_model)

        x = self.dropout(x)
        logits = self.classifier(x)        # (batch, num_classes)
        return logits


def build_model(cfg_model: dict) -> nn.Module:
    """
    Build TransformerSequenceClassifier based on the model config.

    Expected keys in cfg_model:
      - d_model
      - n_heads
      - num_layers
      - dim_feedforward
      - dropout
      - vocab_size
      - num_classes
      - max_seq_len
    """
    d_model = cfg_model.get("d_model", 128)
    n_heads = cfg_model.get("n_heads", 4)
    num_layers = cfg_model.get("num_layers", 2)
    dim_feedforward = cfg_model.get("dim_feedforward", 256)
    dropout = cfg_model.get("dropout", 0.1)
    vocab_size = cfg_model.get("vocab_size", 1000)
    num_classes = cfg_model.get("num_classes", 2)
    max_seq_len = cfg_model.get("max_seq_len", 64)

    model = TransformerSequenceClassifier(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        num_classes=num_classes,
        max_seq_len=max_seq_len,
        dropout=dropout,
    )

    return model
