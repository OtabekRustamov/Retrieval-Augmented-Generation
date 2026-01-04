# Query Transformation Techniques

---
A comprehensive implementation of **6 query transformation techniques** for Retrieval-Augmented Generation (RAG) systems.

Query transformation improves retrieval by modifying the user's question before searching. Instead of using the raw query directly, transformation techniques generate better search queries that retrieve more relevant documents.

This project implements **6 transformation techniques**, each solving a different problem:

| Technique | Problem Solved | LLM Calls |
|-----------|---------------|-----------|
| **Multi-Query** | Ambiguous queries | 2 |
| **RAG-Fusion** | Poor ranking quality | 2 |
| **HyDE** | Semantic gap between Q&A | 2 |
| **Decomposition** | Complex multi-part questions | 3-5 |
| **Step-Back** | Missing background context | 2 |
| **RRR** | Vague/messy queries | 2 |

---
## Transformation Techniques

### Technique 1: Multi-Query
Generates **multiple variations** of the original question to catch different relevant documents.

```
Query → LLM → Variation 1 → Retrieve → Combine
            → Variation 2 → Retrieve →   ↓
            → Variation 3 → Retrieve → Deduplicate → Answer
```

**Use when:** Query is ambiguous or could be phrased multiple ways.

### Technique 2: RAG-Fusion
Multi-Query + **Reciprocal Rank Fusion (RRF)** for intelligent ranking.

```
Query → LLM → Variations → Retrieve Each → RRF Ranking → Top Results → Answer
```

**RRF Formula:** `score = Σ(1 / (k + rank))` where k=60

**Use when:** You need the best possible ranking from multiple query variations.

### Technique 3: HyDE (Hypothetical Document Embeddings)
Generates a **hypothetical answer** first, then embeds it for retrieval.

```
Query → LLM → Hypothetical Answer → Embed → Find Similar Real Docs → Answer
```

**Key insight:** Hypothetical answers are closer in embedding space to real answers (~0.89) than questions (~0.65).

**Use when:** Technical/semantic queries where question-answer gap is large.

### Technique 4: Query Decomposition
Breaks **complex questions into sub-questions**, answers each, then synthesizes.

```
Query → LLM → Sub-Q1 → Retrieve → Answer1 ─┐
            → Sub-Q2 → Retrieve → Answer2 ─┼→ Synthesize → Final Answer
            → Sub-Q3 → Retrieve → Answer3 ─┘
```

**Use when:** Complex multi-part questions that need step-by-step answers.

### Technique 5: Step-Back Prompting
Generates a **broader question first** to get background context.

```
Query → LLM → Step-Back Question → Retrieve General Context ─┐
        ↓                                                     ├→ Combine → Answer
        └──────────────────────→ Retrieve Specific Context ───┘
```

**Use when:** Specific questions that need conceptual background.

### Technique 6: RRR (Rewrite-Retrieve-Read)
**Rewrites vague queries** into clear, searchable ones before retrieval.

```
Vague Query → LLM Rewrite → Clear Query → Retrieve → Answer
```

**Use when:** User queries are messy, informal, or contain typos.

---

##  Add your documents
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
from app import multi_query_rag, rag_fusion_rag, hyde_rag

# Multi-Query
answer, queries, docs = multi_query_rag("Where did Alex study?")

# RAG-Fusion
answer, ranked_docs = rag_fusion_rag("What optimization was used?")

# HyDE
answer, hypothetical, docs = hyde_rag("How does the model work?")
```
---
##  Technique Details

### Technique 1: Multi-Query

Generates 3-5 variations of the original question.

```python
from app import multi_query_rag

# Ambiguous query → multiple perspectives
answer, queries, docs = multi_query_rag("Where did Alex study?")
# Generated queries:
#   1. "What university did Alex attend?"
#   2. "Where did Alex receive his education?"
#   3. "What is Alex's educational background?"
```

### Technique 2: RAG-Fusion

Multi-Query + RRF ranking for best results.

```python
from app import rag_fusion_rag

answer, docs = rag_fusion_rag("What optimization technique was used?")
# Documents ranked by RRF score across all query variations
# RRF ensures documents appearing highly in multiple queries rank first
```

### Technique 3: HyDE

Generate hypothetical answer, embed it, find similar real docs.

```python
from app import hyde_rag

answer, hypothetical, docs = hyde_rag("How does BERT handle context?")
# Hypothetical: "BERT uses bidirectional transformers to process..."
# This hypothetical is embedded and used to find real documents
```

### Technique 4: Query Decomposition

Break complex questions into sub-questions.

```python
from app import decomposition_rag

answer, sub_answers = decomposition_rag(
    "Compare the model architecture with training procedure"
)
# Sub-questions:
#   1. "What is the model architecture?"
#   2. "What is the training procedure?"
#   3. "How do they compare?"
```

### Technique 5: Step-Back Prompting

Get background context with a broader question.

```python
from app import step_back_rag

answer, step_back_q, docs = step_back_rag(
    "What optimizer did the RL model use?"
)
# Step-back: "What are common optimizers in reinforcement learning?"
# Retrieves both general context and specific answer
```

### Technique 6: RRR

Rewrite vague queries into clear ones.

```python
from app import rrr_rag

answer, rewritten, docs = rrr_rag("that molecule optimization stuff")
# Rewritten: "molecular optimization techniques and methods"
```
---
##  Configuration

All configuration is centralized in `config.py`:

```python
# Paths
DATA_PATH = "D:/LLM/RAG/data/"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Models (OpenAI)
model = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Available functions
setup_vectorstore()       # Create/load vectorstore
format_docs(docs)         # Format documents to string
get_retriever(vs, k=5)    # Get retriever from vectorstore
```
---
##  Comparison

| Aspect | Multi-Query | RAG-Fusion | HyDE | Decomposition | Step-Back | RRR |
|--------|------------|-----------|------|---------------|----------|-----|
| **Solves** | Ambiguity | Ranking | Semantic gap | Complexity | Context | Vagueness |
| **LLM Calls** | 2 | 2 | 2 | 3-5 | 2 | 2 |
| **Speed** |  Fast |  Fast |  Fast |  Slow |  Fast |  Fast |
| **Best for** | Ambiguous Q | Best ranking | Technical Q | Multi-part Q | Conceptual Q | Messy input |
---
### Decision Tree

```
What's the problem with retrieval?
├── Query is ambiguous/unclear
│   └── Use Multi-Query or RAG-Fusion
├── Question-answer semantic gap
│   └── Use HyDE
├── Complex multi-part question
│   └── Use Decomposition
├── Needs background context
│   └── Use Step-Back
├── Query is vague/messy
│   └── Use RRR
└── Not sure
    └── Start with Multi-Query (most general)
```

---



