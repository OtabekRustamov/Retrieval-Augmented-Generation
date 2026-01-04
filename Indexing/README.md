# RAG Indexing Techniques

---
A comprehensive implementation of **5 advanced indexing techniques** for Retrieval-Augmented Generation (RAG) systems.

Indexing is crucial for RAG performance. Instead of basic fixed-size chunking, advanced indexing strategies optimize how documents are split, stored, and retrieved to improve both precision and context quality.

This project implements **5 indexing techniques**, each solving a different problem:

| Technique | Problem Solved | Complexity |
|-----------|---------------|------------|
| **Semantic Chunking** | Fixed splits break meaning | Low |
| **Parent-Child** | Precision vs context tradeoff | Medium |
| **Multi-Representation** | Long docs hard to search | Medium |
| **ColBERT** | Keywords get averaged away | High |
| **Document Summary** | Need document-level discovery | Medium |

---
## Indexing Techniques

### Technique 1: Semantic Chunking
Splits text **where meaning changes**, not at fixed character counts.

```
Fixed-size:  "The cat sat on the | mat. Dogs are loyal ani | mals..."
Semantic:    "The cat sat on the mat. | Dogs are loyal animals..."
```

**Use when:** Structured documents with clear topic shifts (SOPs, manuals, reports).

### Technique 2: Parent-Child Retriever
Index **small chunks for precise search**, return **large parents for context**.

```
Parent (2000 chars) ─── stored in docstore
    │
    ├── Child 1 (400 chars) → embedded
    ├── Child 2 (400 chars) → embedded  ← Query matches here
    └── Child 3 (400 chars) → embedded
    
Result: Returns full Parent (2000 chars) for complete context
```

**Use when:** Need balance between precise matching and complete context.

### Technique 3: Multi-Representation Indexing
Search on **short summaries**, return **full documents**.

```
Full Document (5000+ chars) ─── stored with UUID
        │
        ▼ LLM generates
    Summary (200 chars) ─── embedded in vectorstore
    
Query matches Summary → Returns Full Document
```

**Use when:** Long documents where you need full context for generation.

### Technique 4: ColBERT (Token-Level Embeddings)
Creates **separate embeddings for each token**, preserving keyword precision.

```
Standard Embedding:
  "learning rate 0.001" → [single 1536-dim vector]
  Problem: "0.001" gets averaged away
  
ColBERT:
  "learning rate 0.001"
      │       │     │
    [v1]    [v2]  [v3]  ← Token-level vectors
  Benefit: "0.001" preserved for exact matching
```

**Use when:** Keyword queries, exact values, scientific terminology.

### Technique 5: Document Summary Indexing
Index **document summaries** for high-level discovery.

```
Document → LLM Summary → Embed Summary → Store with link to original

Query → Match Summary → Return Summary or Full Document
```

**Use when:** Document-level retrieval, large collections, discovery phase.


---
###  Add your documents
Place your PDF files in `D:/LLM/RAG/data/` (or update `DATA_PATH` in `config.py`)



### Run Streamlit demo:
```bash
streamlit run app_run.py
```

### Run HuggingFace version (no API key):
```bash
python app.py
```

### Use in your code:
```python
from app import semantic_chunking, parent_child_indexing, multi_representation_indexing

# Semantic Chunking
index = semantic_chunking(documents)
docs = index["vectorstore"].similarity_search("query", k=3)

# Parent-Child
index = parent_child_indexing(documents)
parents = retrieve_parent_child("query", index, k=3)

# Multi-Representation
index = multi_representation_indexing(documents)
full_docs = retrieve_multi_rep("query", index, k=3)
```
---
##  Technique Details

### Technique 1: Semantic Chunking

Split at meaning boundaries using embedding similarity.

```python
from app import semantic_chunking

index = semantic_chunking(documents, threshold=95)
# Higher threshold → fewer splits → larger chunks
# Lower threshold → more splits → finer granularity

docs = index["vectorstore"].similarity_search("What is DeMask?", k=3)
```

### Technique 2: Parent-Child Retriever

Small chunks for search, large parents for context.

```python
from app import parent_child_indexing, retrieve_parent_child

index = parent_child_indexing(documents)
# Creates: parent_store (2000 char chunks) + vectorstore (400 char children)

parents = retrieve_parent_child("What optimization was used?", index, k=3)
# Returns full parent chunks with complete context
```

### Technique 3: Multi-Representation Indexing

Search summaries, return full documents.

```python
from app import multi_representation_indexing, retrieve_multi_rep

index = multi_representation_indexing(documents, max_docs=10)
# Creates: doc_store (full docs) + vectorstore (summaries)

full_docs = retrieve_multi_rep("Explain the model architecture", index, k=3)
# Returns complete original documents, not just summaries
```

### Technique 4: ColBERT

Token-level embeddings for keyword precision.

```python
from app import colbert_indexing, colbert_search

index = colbert_indexing(documents, index_name="my_index")
# Requires: pip install ragatouille

results = colbert_search("learning rate 0.001", index, k=3)
# Excellent for exact value matching
```

### Technique 5: Document Summary Indexing

Index summaries for document discovery.

```python
from app import document_summary_indexing, retrieve_doc_summary

index = document_summary_indexing(documents, max_docs=10)

# Return summaries only
summaries = retrieve_doc_summary("main research topic", index, return_original=False)

# Return full documents
full_docs = retrieve_doc_summary("main research topic", index, return_original=True)
```
---
##  Configuration

All configuration is centralized in `config.py`:

```python
# Paths
DATA_PATH = "D:/LLM/RAG/data/"
VECTOR_STORE_PATH = "faiss_index"

# Models (OpenAI)
model = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Chunk sizes
PARENT_CHUNK_SIZE = 2000
CHILD_CHUNK_SIZE = 400

# Available functions
load_documents()          # Load PDFs from DATA_PATH
setup_vectorstore()       # Create basic vectorstore
format_docs(docs)         # Format documents to string
```
---
##  Comparison

| Aspect | Semantic | Parent-Child | Multi-Rep | ColBERT | Doc Summary |
|--------|---------|-------------|-----------|---------|-------------|
| **Solves** | Bad splits | Precision vs context | Long docs | Keywords lost | Discovery |
| **Complexity** | Low | Medium | Medium | High | Medium |
| **LLM Needed** | No | No | Yes | No | Yes |
| **Speed** |  Fast |  Fast |  Slow |  Slow |  Slow |
| **Best for** | Structured docs | General use | Long docs | Exact values | Large collections |
---
### Decision Tree

```
What's the main challenge?
├── Fixed-size chunks break sentences/topics
│   └── Use Semantic Chunking
├── Need both precise search AND full context
│   └── Use Parent-Child
├── Documents are very long, need full context
│   └── Use Multi-Representation
├── Need exact keyword/value matching
│   └── Use ColBERT
├── Need document-level discovery
│   └── Use Document Summary
└── Not sure / General purpose
    └── Start with Parent-Child (best balance)
```

---



