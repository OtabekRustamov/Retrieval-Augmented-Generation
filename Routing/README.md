#  RAG Routing Patterns

---
A comprehensive implementation of **4 query routing patterns** for Retrieval-Augmented Generation (RAG) systems.


Query routing is essential for efficient RAG systems. Instead of searching all data sources for every query, routing directs each query to the most appropriate source or processing method.

This project implements **4 routing patterns**, each answering a different question:

| Pattern | Question | Routes To |
|---------|----------|-----------|
| **Data Source** | WHERE to get data? | Documents / Database / API / LLM |
| **Component** | HOW to process? | Agent / VectorStore / LLM |
| **Prompt Template** | WHAT style? | Technical / Creative / Educational |
| **Agentic** | Agent decides! | Multiple tools dynamically |

---
##  Routing Patterns

### Pattern 1: Data Source Routing
Routes queries to different **data sources** based on query content.

```
Query → Router → Documents (Semantic Search)
              → Database (Text-to-SQL)
              → APIs (Function Calling)
              → General LLM
```

**Use when:** You have multiple data sources (docs, DB, APIs) and need to route to the right one.

### Pattern 2: Component Routing
Routes queries to different **processing components** based on complexity.

```
Query → Router → Agent (complex reasoning)
              → VectorStore (document retrieval)
              → LLM (simple Q&A)
```

**Use when:** Queries vary in complexity - some need simple answers, others need multi-step reasoning.

### Pattern 3: Prompt Template Routing
Routes queries to different **prompt templates** for different response styles.

```
Query → Router → Prompt 1 (Technical) ─┐
              → Prompt 2 (Creative)  ──┼── → LLM → Response
              → Prompt 3 (Educational)─┘
```

**Use when:** You want different response styles (technical vs creative vs educational).

### Pattern 4: Agentic Routing
**Agent dynamically decides** which tools to use - can use multiple!

```
Query → Agent → [Tool 1] → [Tool 2] → ... → Synthesize → Response
```

**Use when:** Complex queries that may need multiple sources or multi-step reasoning.

---

###  Add your documents
Place your PDF files in `D:/LLM/RAG/data/` (or update `DATA_PATH` in `config.py`):
- `CV.pdf` - Your CV/Resume
- `DMS.pdf` - Research documents
- `LLM_Interview.pdf` - Interview prep materials

### Run Streamlit demo:
```bash
streamlit run app_run.py
```

### Run HuggingFace version (no API key):
```bash
python app.py
```
---

##  Pattern Details

### Pattern 1: Data Source Routing

Routes to different data sources based on query type.

```python
from 1_datasource_routing import query_rag, setup_vectorstores

vectorstores = setup_vectorstores()

# Document query → routes to vectorstore
query_rag("What does the CV say about education?", vectorstores)
# Output: datasource="documents"

# Database query → routes to SQL generation
query_rag("How many orders last month?", vectorstores)
# Output: datasource="database"

# API query → routes to function calling
query_rag("Get leads from Salesforce", vectorstores)
# Output: datasource="api"

# General query → routes to LLM
query_rag("What is machine learning?", vectorstores)
# Output: datasource="general_llm"
```

### Pattern 2: Component Routing

Routes to different processing components based on complexity.

```python
from 2_component_routing import query_rag, setup_vectorstores

vectorstores = setup_vectorstores()

# Complex query → routes to agent
query_rag("Calculate compound interest and compare options", vectorstores)
# Output: component="agent"

# Retrieval query → routes to vectorstore
query_rag("Where did Otabek study?", vectorstores)
# Output: component="vectorstore"

# Simple query → routes to LLM
query_rag("What is Python?", vectorstores)
# Output: component="llm"
```

### Pattern 3: Prompt Template Routing

Routes to different prompt styles.

```python
from 3_prompt_routing import query_rag

# Technical question → technical prompt
query_rag("How to implement REST API in Python?")
# Output: prompt_type="technical"

# Creative request → creative prompt
query_rag("Write a poem about coding")
# Output: prompt_type="creative"

# Learning question → educational prompt
query_rag("Explain neural networks to a beginner")
# Output: prompt_type="educational"
```

### Pattern 4: Agentic Routing

Agent dynamically selects and uses multiple tools.

```python
from 4_agentic_routing import query_rag, setup_vectorstores

vectorstores = setup_vectorstores()

# Agent decides which tools to use
result = query_rag("Compare CV skills with interview topics", vectorstores)
# Output: tools_used=["search_cv", "search_llm_interview"]

# Agent can use calculator + search
result = query_rag("Calculate 5% of 10000 and tell me about DMS", vectorstores)
# Output: tools_used=["calculator", "search_dms"]
```
---
##  Configuration

All configuration is centralized in `config.py`:

```python
# Paths
DATA_PATH = "D:/LLM/RAG/data/"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Document mapping
DOC_MAPPING = {
    "cv_resume": "CV.pdf",
    "dms_info": "DMS.pdf",
    "llm_interview": "LLM_Interview.pdf"
}

# Available functions
setup_vectorstores()      # Create/load vectorstores
format_docs(docs)         # Format documents to string
get_retriever(vs, k=4)    # Get retriever from vectorstore
generate_answer(q, ctx)   # Generate answer with context
```
---
##  Comparison

| Aspect | Data Source | Component | Prompt | Agentic |
|--------|------------|----------|-------|---------|
| **Routes to** | Data sources | Processing types | Prompt styles | Multiple tools |
| **# of routes** | ONE | ONE | ONE | MULTIPLE |
| **Speed** |  Fast |  Fast |  Fast |  Slow |
| **Cost** |  Low |  Low |  Low |  High |
| **Flexibility** | Medium | Medium | Medium | Very High |
| **Best for** | Multi-source data | Varying complexity | Style adaptation | Complex queries |

### Decision Tree

```
Is query simple with clear category?
├── YES → Use Logical Routing (Pattern 1, 2, or 3)
│         ├── Different data sources? → Pattern 1
│         ├── Different complexity? → Pattern 2
│         └── Different styles? → Pattern 3
└── NO → Use Agentic Routing (Pattern 4)
```


---

