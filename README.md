# GPT Implementation from Scratch

A **step-by-step PyTorch implementation of a GPT-style language model**, built incrementally from a simple bigram model up to a full Transformer decoder with multi-head self-attention. The model is trained on a custom song-lyrics dataset and run on Google Colab with GPU (CUDA) support.

---

## 📖 Project Overview

This project is a learning-oriented implementation that walks through the core ideas behind GPT (Generative Pre-trained Transformer). Each iteration adds a new architectural component, so you can see exactly how and why each piece improves generation quality.

The model is trained to generate text in a **song-lyrics style**, given a starting prompt like `"I used to rule the world"` or `"And it was all yellow"`.

---

## 🏗️ Architecture Evolution

The notebook builds the model incrementally across the following stages:

| Stage | Model | Params |
|-------|-------|--------|
| 1 | **Bigram Model** — token embedding table only | ~5,625 |
| 2 | **Embedding + Linear Layer** — separate embedding & projection | ~19,275 |
| 3 | **+ Positional Embeddings** — position-aware token representations | ~27,467 |
| 4 | **+ Single Self-Attention Head** — causal masked attention | ~76,619 |
| 5 | **+ Multi-Head Self-Attention** (8 heads) | larger |
| 6 | **+ FeedForward Layer** — per-token MLP after attention | larger |
| 7 | **Full Decoder Blocks** — stacked `DecoderBlock` (attention + FFN + LayerNorm + Dropout + residual connections) | largest |

The final model is a proper **Transformer decoder** composed of:
- Token embedding table
- Positional embedding table
- Dropout on embeddings
- Stack of `DecoderBlock`s, each containing:
  - `MultiHeadSelfAttention` (causal / masked)
  - `FeedForward` MLP
  - `LayerNorm` (pre-norm residual connections)
- Final `LayerNorm`
- Linear projection head → vocabulary logits

---

## 🔑 Key Concepts Implemented

- **Character-level tokenization** — custom `encode` / `decode` functions over a 75-character vocabulary
- **GPTDataLoader** — sequential, wrap-around batch sampler
- **Causal self-attention** — lower-triangular mask prevents tokens from attending to the future
- **Three attention weight implementations** compared:
  - Naive loop (version 1)
  - Lower-triangular matrix division (version 2)
  - Softmax + masked fill (version 3 — used in production)
- **Multi-head self-attention** — parallel attention heads concatenated and projected
- **Positional embeddings** — learned, added to token embeddings
- **Residual connections** — `x = x + sublayer(x)` for stable deep training
- **Layer normalization** — applied before each sub-layer (pre-norm)
- **Dropout** — regularization in attention output and embeddings
- **Advanced generation strategies** — temperature scaling, top-k filtering, top-p (nucleus) sampling
- **Checkpoint saving & resuming** — periodic saves to Google Drive

---

## ⚙️ Hyperparameters

| Parameter | Value |
|-----------|-------|
| Training batch size | 128 |
| Eval batch size | 64 |
| Context length | 64 tokens |
| Embedding size | 128 |
| Number of attention heads | 4–8 |
| Head size | `embedding_size / num_heads` |
| Train / eval split | 80% / 20% |
| Optimizer | AdamW (lr = 1e-3) |
| Dropout | 0.1 |
| Epochs per stage | 50 |

---

## 🚀 Getting Started

### Requirements

```bash
pip install torch
```

### Dataset

Place `songs_final_cleaned.txt` in your working directory (or Google Drive under `/MyDrive/GPT/`). The file contains ~903,785 characters of song lyrics used as the training corpus.

### Running

The project is designed to run in **Google Colab** with a GPU runtime:

1. Open `GPT_implementation_from_scratch.ipynb` in Colab.
2. Mount your Google Drive (for checkpointing).
3. Run cells top-to-bottom — each section builds on the previous one.

You can also run locally with any CUDA-capable GPU. The notebook automatically falls back to CPU if no GPU is found.

---

## 📊 Training Results (Sample)

Training loss drops significantly once self-attention is introduced:

| Model | Train Loss (epoch 50) |
|-------|-----------------------|
| Bigram | ~2.33 |
| + Linear layer | ~2.36 |
| + Positional embedding | ~2.37 |
| + Single self-attention head | ~2.13 |

---

## 💡 Inspiration

This implementation is inspired by Andrej Karpathy's [*"Let's build GPT"*](https://www.youtube.com/watch?v=kCc8FmEb1nY) lecture, following the same incremental build-up from bigrams to Transformers.

---

## 📄 License

This project is for educational purposes.
