"""
============================================================================
CONFIG - Shared Configuration for Query Transformation Techniques
============================================================================
"""

import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader, YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS

# ============================================================================
# API KEY - Replace with your key or use environment variable
# ============================================================================
# Option 1: Direct (not recommended for production)
# API_KEY = "your-api-key-here"

# Option 2: From file
from sk import my_gpt as API_KEY

# Option 3: From environment variable
# API_KEY = os.getenv("OPENAI_API_KEY")


# ============================================================================
# PATHS
# ============================================================================
DATA_PATH = "D:/LLM/RAG/data/"
VECTOR_STORE_PATH = "faiss_index"

YOUTUBE_URLS = [
    "https://www.youtube.com/watch?v=4STrDD7leEM",
    "https://www.youtube.com/watch?v=p2_cRrKScvE",
]


# ============================================================================
# MODELS
# ============================================================================
model = ChatOpenAI(
    api_key=API_KEY, 
    model="gpt-4o-mini", 
    temperature=0
)

embeddings = OpenAIEmbeddings(
    api_key=API_KEY, 
    model="text-embedding-3-small"
)


# ============================================================================
# TEXT SPLITTERS
# ============================================================================
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200
)

semantic_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95
)


# ============================================================================
# VECTORSTORE SETUP
# ============================================================================
def setup_vectorstore(force_rebuild=False):
    """Load or create vectorstore with PDF and YouTube sources"""
    
    if os.path.exists(VECTOR_STORE_PATH) and not force_rebuild:
        print("Loading existing vector store...")
        vectorstore = FAISS.load_local(
            VECTOR_STORE_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        print("Vector store loaded!\n")
        return vectorstore
    
    print("Creating new vector store...")
    all_chunks = []
    
    # Load PDFs
    print("Loading PDFs...")
    pdf_docs = []
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(DATA_PATH, file))
            loaded_docs = loader.load()
            
            for doc in loaded_docs:
                doc.metadata["source_type"] = "pdf"
                doc.metadata["filename"] = file
            
            pdf_docs.extend(loaded_docs)
            print(f"  Loaded: {file} ({len(loaded_docs)} pages)")
    
    # Semantic chunking for PDFs
    if pdf_docs:
        pdf_chunks = semantic_splitter.split_documents(pdf_docs)
        all_chunks.extend(pdf_chunks)
        print(f"  Created {len(pdf_chunks)} semantic chunks from PDFs")
    
    # Load YouTube
    if YOUTUBE_URLS:
        print("Loading YouTube...")
        youtube_docs = []
        
        for url in YOUTUBE_URLS:
            try:
                clean_url = url.split("&")[0]
                loader = YoutubeLoader.from_youtube_url(
                    clean_url,
                    add_video_info=False,
                    language=["en"]
                )
                loaded_docs = loader.load()
                
                for doc in loaded_docs:
                    doc.metadata["source_type"] = "youtube"
                    doc.metadata["source_url"] = clean_url
                
                youtube_docs.extend(loaded_docs)
                print(f"  Loaded: {clean_url[-25:]}")
            except Exception as e:
                print(f"  Failed: {clean_url[-25:]} - {e}")
        
        # Recursive chunking for YouTube
        if youtube_docs:
            youtube_chunks = recursive_splitter.split_documents(youtube_docs)
            all_chunks.extend(youtube_chunks)
            print(f"  Created {len(youtube_chunks)} chunks from YouTube")
    
    # Create and save vectorstore
    print(f"\nTotal chunks: {len(all_chunks)}")
    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    vectorstore.save_local(VECTOR_STORE_PATH)
    print(f"Vector store saved to {VECTOR_STORE_PATH}\n")
    
    return vectorstore


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def format_docs(docs):
    """Format documents into a single context string"""
    return "\n\n".join([doc.page_content for doc in docs])


def get_retriever(vectorstore, k=5):
    """Get a retriever from vectorstore"""
    return vectorstore.as_retriever(search_kwargs={"k": k})
