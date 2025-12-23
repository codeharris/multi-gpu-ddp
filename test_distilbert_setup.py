"""
Quick test to verify DistilBERT setup is working correctly.
Run this before submitting SLURM jobs.
"""

import torch
from transformers import DistilBertTokenizer, DistilBertModel

print("=" * 50)
print("Testing DistilBERT Setup")
print("=" * 50)

# Test 1: Load tokenizer
print("\n1. Loading tokenizer...")
try:
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    print("✓ Tokenizer loaded successfully")
except Exception as e:
    print(f"✗ Failed to load tokenizer: {e}")
    exit(1)

# Test 2: Load model
print("\n2. Loading DistilBERT model...")
try:
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    print("✓ Model loaded successfully")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,} (~{total_params/1e6:.1f}M)")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    exit(1)

# Test 3: Tokenize sample text
print("\n3. Testing tokenization...")
try:
    text = "This is a test sentence for AG News classification."
    encoded = tokenizer(text, max_length=256, padding='max_length', truncation=True, return_tensors='pt')
    print(f"✓ Tokenized successfully")
    print(f"  Input shape: {encoded['input_ids'].shape}")
    print(f"  Attention mask shape: {encoded['attention_mask'].shape}")
except Exception as e:
    print(f"✗ Failed to tokenize: {e}")
    exit(1)

# Test 4: Forward pass
print("\n4. Testing forward pass...")
try:
    with torch.no_grad():
        outputs = model(**encoded)
    print(f"✓ Forward pass successful")
    print(f"  Output shape: {outputs.last_hidden_state.shape}")
except Exception as e:
    print(f"✗ Failed forward pass: {e}")
    exit(1)

# Test 5: Import custom model
print("\n5. Testing custom DistilBERT classifier...")
try:
    import sys
    sys.path.insert(0, 'src')
    from models.distilbert_model import DistilBertClassifier
    
    classifier = DistilBertClassifier(num_classes=4, pretrained=False)
    total, trainable = classifier.count_parameters()
    print(f"✓ Custom classifier loaded")
    print(f"  Total: {total:,} | Trainable: {trainable:,}")
except Exception as e:
    print(f"✗ Failed to import custom model: {e}")
    exit(1)

# Test 6: GPU availability
print("\n6. Checking GPU...")
if torch.cuda.is_available():
    print(f"✓ CUDA available")
    print(f"  Device count: {torch.cuda.device_count()}")
    print(f"  Current device: {torch.cuda.current_device()}")
    print(f"  Device name: {torch.cuda.get_device_name(0)}")
else:
    print("⚠ CUDA not available (CPU mode)")

print("\n" + "=" * 50)
print("All tests passed! Ready to run experiments.")
print("=" * 50)
print("\nNext steps:")
print("  Single GPU: sbatch scripts/run_agnews_distilbert_single_gpu.sh")
print("  4 GPUs DDP: sbatch scripts/run_agnews_distilbert_ddp_4gpu.sh")
