"""
DistilBERT-based text classification model for distributed training.
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig


class DistilBertClassifier(nn.Module):
    """
    DistilBERT model for text classification.
    Uses pre-trained DistilBERT with a classification head.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pre-trained weights
        dropout: Dropout probability for classification head
    """
    
    def __init__(self, num_classes: int, pretrained: bool = True, dropout: float = 0.1):
        super().__init__()
        
        if pretrained:
            # Load pre-trained DistilBERT
            self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        else:
            # Initialize from scratch (for comparison)
            config = DistilBertConfig()
            self.distilbert = DistilBertModel(config)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_classes)
        
        # Initialize classifier weights
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass through DistilBERT and classification head.
        
        Args:
            input_ids: Token IDs (batch_size, seq_length)
            attention_mask: Attention mask (batch_size, seq_length)
        
        Returns:
            logits: Classification logits (batch_size, num_classes)
        """
        # Get DistilBERT outputs
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Apply classification head
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        
        return logits
    
    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
    
    def freeze_base_model(self):
        """Freeze DistilBERT parameters (only train classifier)."""
        for param in self.distilbert.parameters():
            param.requires_grad = False
    
    def unfreeze_base_model(self):
        """Unfreeze DistilBERT parameters (train entire model)."""
        for param in self.distilbert.parameters():
            param.requires_grad = True
