# AI Research Assistant

An end-to-end Retrieval-Augmented Generation (RAG) system that fetches live research papers from ArXiv, indexes them with state-of-the-art embeddings, and answers natural-language research queries using a quantized LLM — all wrapped in a Gradio web interface.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Known Limitations](#known-limitations)

---

## Overview

The AI Research Assistant automates literature discovery and synthesis across six research domains. Given a natural-language question, it retrieves the most relevant papers from a live ArXiv corpus, re-ranks them with a cross-encoder for precision, and generates a structured summary using a quantized Qwen2.5-7B model. It also accepts uploaded PDF papers for direct analysis.

---

## Features

- **Live ArXiv ingestion** — fetches up to 50 papers per domain (300 total) at runtime
- **High-quality embeddings** — uses `BAAI/bge-large-en-v1.5`, a top performer on the MTEB retrieval benchmark
- **Two-stage retrieval** — fast FAISS vector search followed by cross-encoder re-ranking
- **4-bit quantized LLM** — `Qwen2.5-7B-Instruct` loaded with NF4 quantization via BitsAndBytes for GPU efficiency; falls back to `distilgpt2` if unavailable
- **PDF analysis** — upload any research paper and receive an AI-generated summary
- **Multi-tab Gradio UI** — separate tabs for research queries, a conversational chatbot, and PDF upload
- **Built-in evaluation suite** — retrieval accuracy, semantic similarity, and ROUGE scores computed automatically

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│  ArXiv Corpus (300 papers)      │
│  6 domains × 50 papers          │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  BAAI/bge-large-en-v1.5         │
│  Sentence Embeddings + FAISS    │
│  (Top-15 candidates retrieved)  │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Cross-Encoder Re-ranker        │
│  ms-marco-MiniLM-L-6-v2         │
│  (Top-5 papers selected)        │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Qwen2.5-7B-Instruct (4-bit)    │
│  RAG Prompt → Answer Generation │
└────────────────┬────────────────┘
                 │
                 ▼
          Gradio Web UI
```

---

## Requirements

- Python 3.9+
- CUDA-capable GPU (recommended; CPU fallback available)
- ~10 GB GPU VRAM for 4-bit Qwen2.5-7B (or CPU-only with `distilgpt2` fallback)

### Python Dependencies

```
bitsandbytes
accelerate
transformers
sentence-transformers
faiss-cpu
arxiv
gradio
PyPDF2
nltk
rouge-score
torch
numpy
```

---

## Installation

1. Clone or download this repository and open the notebook in Jupyter or Google Colab.

2. Run **Cell 1** to install all dependencies:

```bash
pip install -q -U bitsandbytes accelerate transformers sentence-transformers \
    faiss-cpu arxiv gradio pypdf2 nltk rouge-score
```

3. Ensure a Hugging Face token is available in your environment if accessing gated models:

```bash
huggingface-cli login
```

---

## Usage

Run the notebook cells in order:

| Cell | Purpose |
|------|---------|
| 1 | Install dependencies |
| 2 | Imports, device setup |
| 3 | Fetch 300 ArXiv papers across 6 domains |
| 4 | Build FAISS index with BGE embeddings |
| 5 | Load cross-encoder re-ranker |
| 6 | Load Qwen2.5-7B-Instruct (4-bit) or distilgpt2 fallback |
| 7 | Define RAG generation and PDF analysis functions |
| 8–10 | Run evaluation suite |
| 11 | Print final metrics report |
| 12 | Launch Gradio app |

After Cell 12 executes, a public shareable Gradio link is printed. The interface has three tabs:

- **Research Query & Analysis** — enter a free-text question; receive an AI summary and a table of the top retrieved papers with relevance scores
- **Research Chatbot** — conversational interface for follow-up questions
- **Upload PDF Paper** — upload a `.pdf` file to receive a structured summary of its findings, gaps, and future directions

---

## Evaluation

Three automatic metrics are computed in Cells 8–10:

### 1. Retrieval Accuracy (Semantic)
Ten domain-specific queries are evaluated. A result is counted as a hit if the cosine similarity between the expected topic embedding and any retrieved paper exceeds a threshold of 0.40.

### 2. Semantic Similarity
Six domain-spanning queries are answered by the RAG pipeline. The cosine similarity between the query embedding and the generated response embedding is averaged.

| Rating | Score Range |
|--------|------------|
| STRONG | ≥ 0.70 |
| GOOD | 0.50 – 0.69 |
| NEEDS IMPROVEMENT | < 0.50 |

### 3. ROUGE Scores
Four queries with human-written reference answers are used to compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores against the model-generated responses.

Cell 11 prints a consolidated report:

```
FINAL EVALUATION REPORT — AI Research Assistant v2
  Retrieval Accuracy (Semantic) : XX.XX%
  Avg Semantic Similarity       : X.XXXX
  Avg ROUGE-1                   : X.XXXX
  Avg ROUGE-2                   : X.XXXX
  Avg ROUGE-L                   : X.XXXX
```

---

## Project Structure

```
AI_Research_Assistant.ipynb   # Main notebook (all cells)
README.md                     # This file
```

---

## Model Details

| Component | Model | Purpose |
|-----------|-------|---------|
| Embedder | `BAAI/bge-large-en-v1.5` | Dense retrieval embeddings |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Precision re-ranking of candidates |
| Generator | `Qwen/Qwen2.5-7B-Instruct` (4-bit NF4) | Answer generation |
| Fallback Generator | `distilgpt2` | CPU-compatible fallback |

The BGE model requires a query prefix for optimal retrieval:
```
"Represent this sentence for searching relevant passages: <query>"
```
This is applied automatically inside the `retrieve()` function.

---

## Research Domains

The corpus covers the following six domains, with 50 papers each fetched from ArXiv:

- Computer Vision
- Natural Language Processing
- Bioinformatics
- Robotics
- Cybersecurity
- Climate Modeling

---

## Known Limitations

- **Corpus is fetched at runtime** — paper availability depends on the ArXiv API and network access; results may vary between runs.
- **Context window cap** — only the first 3,500 characters of retrieved abstracts are passed to the LLM to stay within the 4,096-token limit.
- **PDF extraction** — only the first 5 pages of uploaded PDFs are analyzed.
- **Fallback model quality** — if Qwen fails to load, `distilgpt2` is used; its outputs are significantly shorter and lower quality.
- **No persistent memory** — the chatbot tab does not maintain conversation state across Gradio sessions.
