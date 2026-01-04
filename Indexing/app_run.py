"""
============================================================================
STREAMLIT APP - RAG INDEXING TECHNIQUES
============================================================================
Run with: streamlit run app_run.py
============================================================================
"""

import streamlit as st
import os
import uuid

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="RAG Indexing Techniques",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_PATH = "D:/LLM/RAG/data/"

# API Key from file or environment
try:
    from sk import my_gpt
    OPENAI_API_KEY = my_gpt
except:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


# ============================================================================
# MODEL INITIALIZATION
# ============================================================================
@st.cache_resource
def get_models():
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0)
    emb = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")
    return llm, emb

llm, embeddings = get_models()


# ============================================================================
# DOCUMENT LOADING
# ============================================================================
@st.cache_data
def load_documents():
    docs = []
    if os.path.exists(DATA_PATH):
        for file in os.listdir(DATA_PATH):
            if file.endswith(".pdf"):
                loader = PyMuPDFLoader(os.path.join(DATA_PATH, file))
                loaded = loader.load()
                for doc in loaded:
                    doc.metadata["filename"] = file
                docs.extend(loaded)
    return docs


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


# ============================================================================
# INDEXING TECHNIQUES
# ============================================================================
def basic_chunking(documents, chunk_size=1000):
    """Basic fixed-size chunking"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return {"vectorstore": vectorstore, "chunks": chunks, "technique": "basic"}


def semantic_chunking(documents):
    """Semantic chunking - split where meaning changes"""
    try:
        from langchain_experimental.text_splitter import SemanticChunker
        splitter = SemanticChunker(embeddings=embeddings, breakpoint_threshold_amount=95)
        chunks = splitter.split_documents(documents)
    except:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)
    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return {"vectorstore": vectorstore, "chunks": chunks, "technique": "semantic"}


def parent_child_indexing(documents):
    """Parent-Child: small chunks for search, return large parents"""
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    
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
            all_children.extend(children)
    
    vectorstore = FAISS.from_documents(all_children, embeddings)
    return {
        "vectorstore": vectorstore,
        "parent_store": parent_store,
        "technique": "parent_child"
    }


def multi_representation(documents, progress_callback=None):
    """Multi-Rep: search summaries, return full docs"""
    doc_store = {}
    summaries = []
    
    max_docs = min(10, len(documents))
    
    for i, doc in enumerate(documents[:max_docs]):
        if progress_callback:
            progress_callback(i + 1, max_docs)
        
        doc_id = str(uuid.uuid4())
        doc_store[doc_id] = doc
        
        template = "Summarize in 2-3 sentences:\n\n{content}\n\nSummary:"
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()
        summary = chain.invoke({"content": doc.page_content[:4000]})
        
        summary_doc = Document(
            page_content=summary,
            metadata={"doc_id": doc_id, "source": doc.metadata.get("filename", "unknown")}
        )
        summaries.append(summary_doc)
    
    vectorstore = FAISS.from_documents(summaries, embeddings)
    return {
        "vectorstore": vectorstore,
        "doc_store": doc_store,
        "summaries": summaries,
        "technique": "multi_rep"
    }


def document_summary(documents, progress_callback=None):
    """Document Summary: index document summaries"""
    summaries = []
    max_docs = min(10, len(documents))
    
    for i, doc in enumerate(documents[:max_docs]):
        if progress_callback:
            progress_callback(i + 1, max_docs)
        
        template = "Summarize in 2-3 sentences:\n\n{content}\n\nSummary:"
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()
        summary = chain.invoke({"content": doc.page_content[:4000]})
        
        summary_doc = Document(
            page_content=summary,
            metadata={
                "doc_index": i,
                "original_content": doc.page_content,
                "source": doc.metadata.get("filename", "unknown")
            }
        )
        summaries.append(summary_doc)
    
    vectorstore = FAISS.from_documents(summaries, embeddings)
    return {
        "vectorstore": vectorstore,
        "summaries": summaries,
        "originals": documents[:max_docs],
        "technique": "doc_summary"
    }


# ============================================================================
# RETRIEVAL FUNCTIONS
# ============================================================================
def retrieve(question, index, k=5):
    """Retrieve documents based on technique"""
    technique = index.get("technique", "basic")
    
    if technique == "parent_child":
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
        return parents, {"child_matches": len(children), "parent_docs": len(parents)}
    
    elif technique == "multi_rep":
        summaries = index["vectorstore"].similarity_search(question, k=k)
        full_docs = []
        for summary in summaries:
            doc_id = summary.metadata.get("doc_id")
            if doc_id and doc_id in index["doc_store"]:
                full_docs.append(index["doc_store"][doc_id])
        return full_docs, {"summary_matches": len(summaries), "full_docs": len(full_docs)}
    
    elif technique == "doc_summary":
        summaries = index["vectorstore"].similarity_search(question, k=k)
        results = []
        for summary in summaries:
            original = summary.metadata.get("original_content", "")
            if original:
                results.append(Document(
                    page_content=original,
                    metadata={"summary": summary.page_content, "source": summary.metadata.get("source")}
                ))
        return results, {"summary_matches": len(summaries)}
    
    else:
        docs = index["vectorstore"].similarity_search(question, k=k)
        return docs, {"chunks": len(docs)}


# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.title("Indexing Techniques")
st.sidebar.markdown("---")

technique = st.sidebar.selectbox(
    "Select Technique",
    [
        "Basic (Fixed-Size)",
        "Semantic Chunking",
        "Parent-Child",
        "Multi-Representation",
        "Document Summary"
    ]
)

technique_info = {
    "Basic (Fixed-Size)": {
        "desc": "Fixed-size chunks with overlap",
        "best_for": "General purpose, baseline"
    },
    "Semantic Chunking": {
        "desc": "Split where meaning changes",
        "best_for": "Structured docs with clear topics"
    },
    "Parent-Child": {
        "desc": "Small chunks search, return large parents",
        "best_for": "Balance precision + context"
    },
    "Multi-Representation": {
        "desc": "Search summaries, return full docs",
        "best_for": "Long documents"
    },
    "Document Summary": {
        "desc": "Index document summaries",
        "best_for": "Document discovery"
    }
}

info = technique_info[technique]
st.sidebar.markdown(f"**What:** {info['desc']}")
st.sidebar.markdown(f"**Best for:** {info['best_for']}")

st.sidebar.markdown("---")
k_results = st.sidebar.slider("Results (k)", 1, 10, 5)
st.sidebar.markdown(f"Data: `{DATA_PATH}`")


# ============================================================================
# MAIN CONTENT
# ============================================================================
st.title("RAG Indexing Techniques")
st.markdown("Compare different document indexing strategies.")

# Load documents
with st.spinner("Loading documents..."):
    documents = load_documents()

if not documents:
    st.error(f"No PDF documents found in {DATA_PATH}")
    st.stop()

st.success(f"Loaded {len(documents)} document pages")

# Question input
st.markdown("### Ask a Question")
question = st.text_input(
    "Enter your question:",
    placeholder="e.g., What optimization technique was used?"
)

# Example buttons
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Main research focus?"):
        st.session_state["q"] = "What is the main research focus?"
with col2:
    if st.button("Optimization used?"):
        st.session_state["q"] = "What optimization technique was used?"
with col3:
    if st.button("Learning rate?"):
        st.session_state["q"] = "What is the learning rate value?"

if "q" in st.session_state:
    question = st.session_state["q"]
    del st.session_state["q"]
    st.rerun()

# Process
if st.button("Index & Retrieve", type="primary", use_container_width=True):
    if question:
        # Create index
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        def update_progress(current, total):
            progress_bar.progress(current / total)
            progress_text.text(f"Processing {current}/{total}...")
        
        with st.spinner(f"Creating {technique} index..."):
            if technique == "Basic (Fixed-Size)":
                index = basic_chunking(documents)
            elif technique == "Semantic Chunking":
                index = semantic_chunking(documents)
            elif technique == "Parent-Child":
                index = parent_child_indexing(documents)
            elif technique == "Multi-Representation":
                index = multi_representation(documents, update_progress)
            else:
                index = document_summary(documents, update_progress)
        
        progress_bar.empty()
        progress_text.empty()
        
        # Retrieve
        with st.spinner("Retrieving..."):
            docs, stats = retrieve(question, index, k_results)
        
        # Generate answer
        with st.spinner("Generating answer..."):
            context = format_docs(docs)
            template = """Answer based on context:

Context:
{context}

Question: {question}

Answer:"""
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | llm | StrOutputParser()
            answer = chain.invoke({"context": context, "question": question})
        
        # Display results
        st.markdown("---")
        st.markdown(f"### Results ({technique})")
        
        # Answer
        st.markdown("#### Answer")
        st.write(answer)
        
        # Stats
        st.markdown("#### Statistics")
        cols = st.columns(len(stats))
        for i, (key, value) in enumerate(stats.items()):
            cols[i].metric(key.replace("_", " ").title(), value)
        
        # Documents
        st.markdown("#### Retrieved Documents")
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", doc.metadata.get("filename", "unknown"))
            with st.expander(f"Document {i+1} - {source} ({len(doc.page_content)} chars)"):
                st.write(doc.page_content[:2000])
                if len(doc.page_content) > 2000:
                    st.write("...")
                
                if "summary" in doc.metadata:
                    st.markdown("**Summary:**")
                    st.info(doc.metadata["summary"])
    else:
        st.warning("Please enter a question!")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "RAG Indexing Techniques - Built with Streamlit + LangChain"
    "</div>",
    unsafe_allow_html=True
)
