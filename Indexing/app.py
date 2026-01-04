"""
============================================================================
RAG INDEXING TECHNIQUES - HuggingFace Version (No API Key Required)
============================================================================
All 5 indexing techniques using local models.

Usage:
    from app import semantic_chunking, parent_child_indexing
    index = semantic_chunking(documents)
============================================================================
"""

import os
import uuid
import re
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ============================================================================
# PATHS
# ============================================================================
DATA_PATH = "D:/LLM/RAG/data/"

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================
print("Initializing models...")

# Try Ollama first, then HuggingFace
try:
    from langchain_ollama import OllamaLLM
    model = OllamaLLM(model="mistral")
    print("Using Ollama (mistral)")
except:
    try:
        from langchain_huggingface import HuggingFacePipeline
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        import torch
        
        model_id = "mistralai/Mistral-7B-Instruct-v0.2"
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=True
        )
        
        pipe = pipeline(
            "text-generation",
            model=hf_model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.1
        )
        model = HuggingFacePipeline(pipeline=pipe)
        print("Using HuggingFace (Mistral-7B)")
    except Exception as e:
        print(f"Model initialization failed: {e}")
        model = None

# Embeddings - always use sentence-transformers
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    print("Using sentence-transformers embeddings")
except:
    embeddings = None
    print("Embeddings initialization failed")


# ============================================================================
# DOCUMENT LOADING
# ============================================================================
def load_documents():
    """Load PDF documents"""
    docs = []
    if not os.path.exists(DATA_PATH):
        print(f"Path not found: {DATA_PATH}")
        return docs
    
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(DATA_PATH, file))
            loaded = loader.load()
            for doc in loaded:
                doc.metadata["filename"] = file
            docs.extend(loaded)
    
    print(f"Loaded {len(docs)} pages")
    return docs


def format_docs(docs):
    """Format documents into context string"""
    return "\n\n".join([doc.page_content for doc in docs])


# ============================================================================
# 1. SEMANTIC CHUNKING
# ============================================================================
def semantic_chunking(documents, threshold=95):
    """
    Split text where meaning changes.
    
    Args:
        documents: List of Document objects
        threshold: Breakpoint threshold (higher = fewer splits)
    
    Returns:
        dict with vectorstore and chunks
    """
    print("\n" + "="*60)
    print("SEMANTIC CHUNKING")
    print("="*60)
    
    try:
        from langchain_experimental.text_splitter import SemanticChunker
        splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=threshold
        )
        
        chunks = []
        for doc in documents:
            if len(doc.page_content) < 100:
                chunks.append(doc)
                continue
            
            try:
                doc_chunks = splitter.split_documents([doc])
                for chunk in doc_chunks:
                    chunk.metadata.update(doc.metadata)
                    chunk.metadata["chunk_method"] = "semantic"
                chunks.extend(doc_chunks)
            except:
                chunks.append(doc)
        
        print(f"Created {len(chunks)} semantic chunks")
        
    except ImportError:
        print("SemanticChunker not available, using fixed-size")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)
    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    return {"vectorstore": vectorstore, "chunks": chunks}


# ============================================================================
# 2. PARENT-CHILD INDEXING
# ============================================================================
def parent_child_indexing(documents, parent_size=2000, child_size=400):
    """
    Index small chunks for search, return large parents for context.
    
    Args:
        documents: List of Document objects
        parent_size: Parent chunk size
        child_size: Child chunk size
    
    Returns:
        dict with vectorstore and parent_store
    """
    print("\n" + "="*60)
    print("PARENT-CHILD INDEXING")
    print("="*60)
    
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_size, chunk_overlap=400
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size, chunk_overlap=50
    )
    
    parent_store = {}
    all_children = []
    
    for doc in documents:
        parents = parent_splitter.split_documents([doc])
        
        for parent in parents:
            parent_id = str(uuid.uuid4())
            parent_store[parent_id] = parent
            
            children = child_splitter.split_documents([parent])
            for child in children:
                child.metadata["parent_id"] = parent_id
                child.metadata["chunk_type"] = "child"
            
            all_children.extend(children)
    
    print(f"Parents: {len(parent_store)}, Children: {len(all_children)}")
    
    vectorstore = FAISS.from_documents(all_children, embeddings)
    
    return {"vectorstore": vectorstore, "parent_store": parent_store}


def retrieve_parent_child(question, index, k=3):
    """Retrieve using parent-child index"""
    children = index["vectorstore"].similarity_search(question, k=k*2)
    
    parents = []
    seen = set()
    
    for child in children:
        parent_id = child.metadata.get("parent_id")
        if parent_id and parent_id not in seen:
            parent = index["parent_store"].get(parent_id)
            if parent:
                parents.append(parent)
                seen.add(parent_id)
        
        if len(parents) >= k:
            break
    
    return parents


# ============================================================================
# 3. MULTI-REPRESENTATION INDEXING
# ============================================================================
def multi_representation_indexing(documents, max_docs=10):
    """
    Search summaries, return full documents.
    
    Args:
        documents: List of Document objects
        max_docs: Maximum documents to process (LLM calls)
    
    Returns:
        dict with vectorstore and doc_store
    """
    print("\n" + "="*60)
    print("MULTI-REPRESENTATION INDEXING")
    print("="*60)
    
    doc_store = {}
    summaries = []
    
    template = """Summarize in 2-3 sentences:

{content}

Summary:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model | StrOutputParser()
    
    for i, doc in enumerate(documents[:max_docs]):
        doc_id = str(uuid.uuid4())
        doc_store[doc_id] = doc
        
        print(f"  [{i+1}/{min(len(documents), max_docs)}] Summarizing...")
        
        content = doc.page_content[:4000]
        summary = chain.invoke({"content": content})
        
        summary_doc = Document(
            page_content=summary,
            metadata={
                "doc_id": doc_id,
                "source": doc.metadata.get("filename", "unknown")
            }
        )
        summaries.append(summary_doc)
    
    print(f"Created {len(summaries)} summaries")
    
    vectorstore = FAISS.from_documents(summaries, embeddings)
    
    return {"vectorstore": vectorstore, "doc_store": doc_store, "summaries": summaries}


def retrieve_multi_rep(question, index, k=3):
    """Retrieve using multi-representation index"""
    summaries = index["vectorstore"].similarity_search(question, k=k)
    
    full_docs = []
    for summary in summaries:
        doc_id = summary.metadata.get("doc_id")
        if doc_id and doc_id in index["doc_store"]:
            full_docs.append(index["doc_store"][doc_id])
    
    return full_docs


# ============================================================================
# 4. COLBERT INDEXING
# ============================================================================
def colbert_indexing(documents, index_name="colbert_index"):
    """
    Token-level embeddings for keyword precision.
    
    Args:
        documents: List of Document objects
        index_name: Name for the ColBERT index
    
    Returns:
        dict with RAG model
    """
    print("\n" + "="*60)
    print("COLBERT INDEXING")
    print("="*60)
    
    try:
        from ragatouille import RAGPretrainedModel
        
        RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
        
        texts = [doc.page_content for doc in documents]
        ids = [f"doc_{i}" for i in range(len(documents))]
        
        print(f"Indexing {len(texts)} documents...")
        RAG.index(
            collection=texts,
            document_ids=ids,
            index_name=index_name,
            max_document_length=512,
            split_documents=True
        )
        
        print("ColBERT index created")
        return {"RAG": RAG, "documents": documents}
        
    except ImportError:
        print("RAGatouille not installed. Install with: pip install ragatouille")
        return None


def colbert_search(question, index, k=3):
    """Search using ColBERT index"""
    if index is None or "RAG" not in index:
        return []
    
    results = index["RAG"].search(question, k=k)
    
    docs = []
    for r in results:
        docs.append(Document(
            page_content=r.get("content", ""),
            metadata={"score": r.get("score", 0)}
        ))
    
    return docs


# ============================================================================
# 5. DOCUMENT SUMMARY INDEXING
# ============================================================================
def document_summary_indexing(documents, max_docs=10):
    """
    Index document summaries for discovery.
    
    Args:
        documents: List of Document objects
        max_docs: Maximum documents to process
    
    Returns:
        dict with vectorstore, summaries, and originals
    """
    print("\n" + "="*60)
    print("DOCUMENT SUMMARY INDEXING")
    print("="*60)
    
    summaries = []
    
    template = """Summarize in 2-3 sentences:

{content}

Summary:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model | StrOutputParser()
    
    for i, doc in enumerate(documents[:max_docs]):
        print(f"  [{i+1}/{min(len(documents), max_docs)}] Summarizing...")
        
        content = doc.page_content[:4000]
        summary = chain.invoke({"content": content})
        
        summary_doc = Document(
            page_content=summary,
            metadata={
                "doc_index": i,
                "original_content": doc.page_content,
                "source": doc.metadata.get("filename", "unknown")
            }
        )
        summaries.append(summary_doc)
    
    print(f"Created {len(summaries)} summaries")
    
    vectorstore = FAISS.from_documents(summaries, embeddings)
    
    return {
        "vectorstore": vectorstore,
        "summaries": summaries,
        "originals": documents[:max_docs]
    }


def retrieve_doc_summary(question, index, k=3, return_original=True):
    """Retrieve using document summary index"""
    summaries = index["vectorstore"].similarity_search(question, k=k)
    
    if return_original:
        results = []
        for summary in summaries:
            original = summary.metadata.get("original_content", "")
            if original:
                results.append(Document(
                    page_content=original,
                    metadata={"summary": summary.page_content}
                ))
            else:
                results.append(summary)
        return results
    
    return summaries


# ============================================================================
# COMPLETE RAG PIPELINE
# ============================================================================
def indexing_rag(question, index, technique="basic", k=3):
    """
    Complete RAG with any indexing technique.
    
    Args:
        question: User question
        index: Index from any technique
        technique: Which technique was used
        k: Number of documents
    
    Returns:
        Generated answer
    """
    # Retrieve based on technique
    if technique == "parent_child":
        docs = retrieve_parent_child(question, index, k)
    elif technique == "multi_rep":
        docs = retrieve_multi_rep(question, index, k)
    elif technique == "colbert":
        docs = colbert_search(question, index, k)
    elif technique == "doc_summary":
        docs = retrieve_doc_summary(question, index, k)
    else:
        docs = index["vectorstore"].similarity_search(question, k=k)
    
    context = format_docs(docs)
    
    template = """Answer based on context:

Context:
{context}

Question: {question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model | StrOutputParser()
    
    answer = chain.invoke({"context": context, "question": question})
    
    return answer, docs


# ============================================================================
# TEST
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("RAG INDEXING TECHNIQUES TEST")
    print("="*60)
    
    documents = load_documents()
    
    if not documents:
        print("No documents found")
        exit()
    
    # Test semantic chunking
    print("\n--- Testing Semantic Chunking ---")
    index = semantic_chunking(documents[:5])
    docs = index["vectorstore"].similarity_search("What is the main topic?", k=2)
    print(f"Retrieved {len(docs)} docs")
    
    # Test parent-child
    print("\n--- Testing Parent-Child ---")
    index = parent_child_indexing(documents[:5])
    docs = retrieve_parent_child("What optimization was used?", index, k=2)
    print(f"Retrieved {len(docs)} parents")
    
    print("\nTest completed!")
