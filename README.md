#  Advanced RAG Techniques

A simple implementation of advanced RAG (Retrieval-Augmented Generation) techniques organized by pipeline stage.

---

## Overview
This repository contains clean, modular implementations of RAG techniques. Each technique is organized by its pipeline stage for easy understanding.

**Key Features:**
- Organized by Pipeline Stage
- Each technique in separate file with detailed comments
- HuggingFace models (no API key required)
- Streamlit apps for interactive demos
- README for each stage

---

## RAG Pipeline Stages
![RAG Pipeline Stages](images/rag.png)


| Stage | What it Does | Techniques |
|-------|--------------|------------|
| **Query Construction** | Convert NL to database queries | Text-to-SQL, Text-to-Cypher, Text-to-DSL |
| **Query Transformation** | Transform queries before retrieval | Multi-Query, RAG-Fusion, HyDE, Decomposition, Step-Back, RRR |
| **Routing** | Select data source/strategy | Logical, Semantic, Agentic |
| **Indexing** | Prepare documents for retrieval | Semantic Chunking, Parent-Child, Multi-Rep, ColBERT, RAPTOR, Sentence Window |
| **Retrieval** | Find relevant documents | Vector Search, BM25, Hybrid, MMR, Self-Query, Ensemble |
| **Post-Retrieval** | Refine retrieved documents | Re-ranking, RRF, Cohere Rerank, LLM Rerank, Filtering, Compression |
| **Generation** | Generate verified answers | Self-RAG, CRAG, Adaptive RAG, RAG + CoT |
| **Evaluation** | Measure performance | RAGAS, DeepEval, TruLens, LangSmith |

---

## Models Used

All `app.py` files use HuggingFace models (free, runs locally):

| Component | Model | Notes |
|-----------|-------|-------|
| **Embeddings** | `sentence-transformers/all-mpnet-base-v2` | Best quality, 768 dim |
| **LLM** | `mistralai/Mistral-7B-Instruct-v0.2` | With 4-bit quantization |
| **Reranker** | `BAAI/bge-reranker-base` | Cross-encoder reranking |

**Quantization Setup (for 8GB GPU):**
```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)
```

**Alternative Models:**
- Low VRAM: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (no quantization needed)
- CPU only: Use `Ollama` with `mistral` or `llama2`
- Embeddings alternative: `sentence-transformers/all-MiniLM-L6-v2` (faster, 384 dim)

---
###  Setup API key
Create `sk.py` in the project root:
```python
my_gpt = "your-openai-api-key-here"
```

## Folder Structure

Each stage folder contains:

```
Stage_Name/
├── 1_technique_name.py    # Individual technique (OpenAI API)
├── 2_technique_name.py    # Individual technique (OpenAI API)
├── ...
├── config.py              # Shared configuration
├── app.py                 # All techniques combined (HuggingFace)
├── app_run.py             # Streamlit application
└── README.md              # Stage documentation
```

- **Individual files (`1_xxx.py`)**: Use OpenAI API for simplicity
- **`app.py`**: Uses HuggingFace models, no API key needed
- **`app_run.py`**: Interactive Streamlit demo

---



## Stage READMEs

Each stage has its own detailed README:

- [Query Construction](./Query_Construction/README.md)
- [Query Transformation](./Query%20Transformation/README.md)
- [Routing](./Routing/README.md)
- [Indexing](Indexing/README.md)
- [Retrieval](./Retrieval/README.md)
- [Post-Retrieval](./Post_Retrieval/README.md)
- [Generation](./Generation/README.md)
- [Evaluation](./Evaluation/README.md)

---
