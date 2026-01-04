"""
============================================================================
CONFIG - Shared Configuration for RAG Indexing Techniques
============================================================================
"""

import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ============================================================================
# API KEY - Choose one method:
# ============================================================================
# Method 1: Direct (not recommended for production)
OPENAI_API_KEY = "your-api-key-here"

# Method 2: From file
# with open("sk.txt", "r") as f:
#     OPENAI_API_KEY = f.read().strip()

# Method 3: From environment
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ============================================================================
# PATHS
# ============================================================================
DATA_PATH = "D:/LLM/RAG/data/"
VECTOR_STORE_PATH = "faiss_index"

# ============================================================================
# MODELS
# ============================================================================
model = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")

# Aliases
llm = model

# ============================================================================
# TEXT SPLITTERS
# ============================================================================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Parent-Child splitters
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=400
)

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)


# ============================================================================
# DOCUMENT LOADING
# ============================================================================
def load_documents():
    """Load PDF documents from DATA_PATH"""
    docs = []
    
    if not os.path.exists(DATA_PATH):
        print(f"Data path not found: {DATA_PATH}")
        return docs
    
    print("Loading PDFs...")
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            filepath = os.path.join(DATA_PATH, file)
            loader = PyMuPDFLoader(filepath)
            loaded = loader.load()
            
            for doc in loaded:
                doc.metadata["filename"] = file
                doc.metadata["source_type"] = "pdf"
            
            docs.extend(loaded)
            print(f"  {file} ({len(loaded)} pages)")
    
    print(f"Total: {len(docs)} pages\n")
    return docs


def load_all_documents(include_youtube=False):
    """Alias for load_documents"""
    return load_documents()


# ============================================================================
# VECTORSTORE SETUP
# ============================================================================
def setup_vectorstore(force_rebuild=False):
    """Create or load basic vectorstore"""
    store_path = f"{VECTOR_STORE_PATH}_basic"
    
    if os.path.exists(store_path) and not force_rebuild:
        print("Loading existing vectorstore...")
        return FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
    
    print("Creating vectorstore...")
    docs = load_documents()
    chunks = text_splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")
    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(store_path)
    
    return vectorstore


def setup_basic_vectorstore(force_rebuild=False):
    """Alias for setup_vectorstore"""
    return setup_vectorstore(force_rebuild)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def format_docs(docs):
    """Format documents into context string"""
    return "\n\n".join([doc.page_content for doc in docs])


def get_retriever(vectorstore, k=5):
    """Get retriever from vectorstore"""
    return vectorstore.as_retriever(search_kwargs={"k": k})


def print_chunks_info(chunks, name="Chunks"):
    """Print chunk statistics"""
    if not chunks:
        print(f"{name}: No chunks")
        return
    
    lengths = [len(c.page_content) for c in chunks]
    print(f"\n{name} Info:")
    print(f"  Total: {len(chunks)}")
    print(f"  Avg length: {sum(lengths)//len(lengths)} chars")
    print(f"  Min: {min(lengths)}, Max: {max(lengths)} chars")
