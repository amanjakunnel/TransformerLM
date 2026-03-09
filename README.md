# Transformer LM from Scratch

Implementation of decoder-only Transformer LM with BPE tokenizer, trained on TinyStories and OpenWebText.

## Features

- Byte-level BPE tokenizer (GPT-2 regex pre-tokenization, special token handling)
- Pre-norm Transformer blocks (RoPE, causal multi-head attention, SiLU FFN)
- Training infrastructure (AdamW, cosine LR w/ warmup, gradient clipping, checkpointing)
- Memory-efficient data loading (mmap uint16 NumPy arrays)
- Parallel pre-tokenization (multiprocessing on endoftext chunks)

## Quickstart

```bash
# Train BPE tokenizer
uv run python basics/train_bpe.py data/tinystories_train.txt --vocab-size 10000

# Encode datasets
uv run python basics/encode_dataset.py tinystories_train.txt vocab.json merges.json

# Train Transformer LM
uv run python basics/train_lm.py --batch-size 64 --context-length 1024
```

## Architecture
Token IDs → Embedding → [Pre-norm Transformer Blocks] → LM Head → Logits
                      ↓
                RMSNorm → Causal MHSA (RoPE) → RMSNorm → FFN

- Tokenizer: UTF-8 BPE, 256 byte vocab + merges, special tokens (endoftext)
- Model: d_model=512, n_heads=8, n_layers=6, ~50M params
- Training: Cross-entropy loss, AdamW (β1=0.9, β2=0.95), 6e-4 LR peak

## Results
| Dataset     | Vocab Size | Compression       | Training Time   |
| ----------- | ---------- | ----------------- | --------------- |
| TinyStories | 10K        | ~2x bytes/token   | <2min tokenizer |
| OpenWebText | 32K        | ~2.5x bytes/token | 12hr tokenizer  |
