# LLM From Scratch

A hands-on implementation of a Large Language Model (GPT-2 style) built from the ground up in Python, following the concepts from Sebastian Raschka's *Build a Large Language Model (From Scratch)*.

## Overview

This project walks through the complete lifecycle of building, training, and fine-tuning an LLM — from raw text tokenization all the way to instruction fine-tuning — implemented as annotated Python scripts.

## Project Structure

| File | Description |
|------|-------------|
| `llm_from_scratch_1_2.py` | Core LLM implementation: tokenization through pretraining |
| `llm_from_scratch_3.py` | Fine-tuning: spam classification and instruction following |
| `gpt_download.py` | Script to download pretrained GPT-2 weights from OpenAI |
| `requirements.txt` | Python dependencies |
| `the-verdict.txt` | Training text sample (Edith Wharton short story) |

## What's Covered

### Part 1 & 2 — Building and Pretraining (`llm_from_scratch_1_2.py`)

**Tokenization**
- Custom regex-based tokenizers (`SimpleTokenizerV1`, `SimpleTokenizerV2`)
- Special tokens: `<|unk|>`, `<|endoftext|>`
- Byte Pair Encoding (BPE) via `tiktoken` (GPT-2 tokenizer)

**Data Pipeline**
- Sliding window input-target pair generation
- PyTorch `DataLoader` with batching

**Embeddings**
- Token embeddings
- Positional embeddings

**Attention Mechanism** (built step by step)
- Simplified attention
- Self-attention with trainable weights
- Causal (masked) attention
- Dropout regularization
- Multi-head attention (stacked and weight-split implementations)

**GPT Model Architecture**
- Dummy GPT model scaffold
- Layer normalization
- GELU activation + feedforward network
- Shortcut (residual) connections
- Transformer blocks
- Full GPT model implementation
- Text generation from output logits

**Training**
- Cross-entropy loss and perplexity
- Train/validation loss tracking
- Full pretraining loop

**Decoding Strategies**
- Temperature scaling
- Top-k sampling
- Combined temperature + top-k

**Model I/O**
- Saving and loading model weights with PyTorch
- Loading pretrained GPT-2 weights from OpenAI

---

### Part 3 — Fine-tuning (`llm_from_scratch_3.py`)

**Spam Classification (Fine-tuning for Classification)**
- SMS Spam Collection dataset download and preprocessing
- Balanced dataset creation
- `SpamDataset` PyTorch class with padding/truncation
- Classification head added on top of GPT backbone
- Loss and accuracy calculation
- Supervised fine-tuning loop
- Inference as a spam classifier

**Instruction Fine-tuning**
- Alpaca-style prompt formatting
- Train/validation/test dataset splitting
- Instruction dataset batching with custom collation
- Loading pretrained GPT-2 weights as base
- Fine-tuning on instruction-response pairs
- Response extraction and saving
- Model evaluation

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- PyTorch
- tiktoken
- NumPy
- matplotlib
- tensorflow (for GPT-2 weight loading)
- tqdm
- certifi

## Running

Each script is a standalone annotated walkthrough. Run them top-to-bottom:

```bash
python llm_from_scratch_1_2.py
python llm_from_scratch_3.py
```

> Note: `llm_from_scratch_1_2.py` trains and saves model weights (`model.pth`, `model_and_optimizer.pth`) that are used by `llm_from_scratch_3.py`.

## Reference

Based on: [Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch) by Sebastian Raschka
