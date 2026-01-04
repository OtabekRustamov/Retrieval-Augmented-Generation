"""
============================================================================
CONFIG.PY - Shared Configuration for RAG Routing Patterns
============================================================================
Central configuration file for all routing techniques.
Handles LLM, embeddings, vectorstore setup, and helper functions.

Compatible with: langchain >= 0.2.0, langchain-openai >= 0.1.0
============================================================================
"""

import os
from typing import List, Dict, Tuple
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================================================================
# API KEY
# ============================================================================
from sk import my_gpt

# ============================================================================
# PATHS & SETTINGS
# ============================================================================
DATA_PATH = "D:/LLM/RAG/data/"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Document mapping - customize these to your PDF files
DOC_MAPPING = {
    "cv_resume": "CV.pdf",
    "dms_info": "DMS.pdf",
    "llm_interview": "LLM_Interview.pdf"
}

# ============================================================================
# LLM & EMBEDDINGS INITIALIZATION
# ============================================================================
llm = ChatOpenAI(
    api_key=my_gpt,
    model="gpt-4o-mini",
    temperature=0
)

embeddings = OpenAIEmbeddings(
    api_key=my_gpt,
    model="text-embedding-3-small"
)


# ============================================================================
# VECTORSTORE SETUP
# ============================================================================
def setup_vectorstores() -> Dict[str, FAISS]:
    """
    Setup vectorstores for each document type.
    Loads from disk if exists, otherwise creates and saves.
    
    Returns:
        Dict mapping doc_type -> FAISS vectorstore
    """
    vectorstores = {}
    
    for doc_type, filename in DOC_MAPPING.items():
        file_path = os.path.join(DATA_PATH, filename)
        vector_path = f"faiss_index_{doc_type}"
        
        # Try to load existing vectorstore
        if os.path.exists(vector_path):
            vectorstores[doc_type] = FAISS.load_local(
                vector_path, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            print(f"✓ Loaded {doc_type}")
        
        # Create new vectorstore from PDF
        elif os.path.exists(file_path):
            loader = PyMuPDFLoader(file_path)
            docs = loader.load()
            
            # Add metadata
            for doc in docs:
                doc.metadata['document_type'] = doc_type
                doc.metadata['source_file'] = filename
            
            # Split into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            chunks = splitter.split_documents(docs)
            
            # Create and save vectorstore
            vectorstores[doc_type] = FAISS.from_documents(chunks, embeddings)
            vectorstores[doc_type].save_local(vector_path)
            print(f"✓ Created {doc_type} ({len(chunks)} chunks)")
        
        else:
            print(f"⚠️  Not found: {filename}")
    
    return vectorstores


def setup_single_vectorstore(pdf_path: str, name: str = "default") -> FAISS:
    """
    Setup a single vectorstore from a PDF file.
    
    Args:
        pdf_path: Path to PDF file
        name: Name for the vectorstore
    
    Returns:
        FAISS vectorstore
    """
    vector_path = f"faiss_index_{name}"
    
    # Load existing
    if os.path.exists(vector_path):
        vs = FAISS.load_local(
            vector_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        print(f"✓ Loaded {name}")
        return vs
    
    # Create new
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(vector_path)
    print(f"✓ Created {name} ({len(chunks)} chunks)")
    
    return vs


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def format_docs(docs: List[Document]) -> str:
    """Format documents into a single context string."""
    return "\n\n".join([doc.page_content for doc in docs])


def get_retriever(vectorstore: FAISS, k: int = 4):
    """Get a retriever from vectorstore."""
    return vectorstore.as_retriever(search_kwargs={"k": k})


def search_all_vectorstores(
    question: str, 
    vectorstores: Dict[str, FAISS], 
    k: int = 2
) -> List[Document]:
    """Search all vectorstores and combine results."""
    all_docs = []
    for name, vs in vectorstores.items():
        docs = vs.similarity_search(question, k=k)
        all_docs.extend(docs)
    return all_docs


def generate_answer(question: str, context: str) -> str:
    """Generate answer using LLM with provided context."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer based ONLY on the context provided.\n\nContext:\n{context}"),
        ("human", "{question}")
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question, "context": context})


# ============================================================================
# PROMPT TEMPLATES (for Prompt Routing - Pattern 3)
# ============================================================================
PROMPT_TEMPLATES = {
    "technical": """You are a technical expert. Your responses should be:
- Precise and accurate with technical terminology
- Include code examples when relevant
- Reference best practices and standards
- Structured and detailed
Focus on accuracy and technical depth.""",

    "creative": """You are a creative writer. Your responses should be:
- Engaging and imaginative
- Use vivid language and metaphors
- Original and expressive
- Artistic and inspiring
Focus on creativity and engagement.""",

    "educational": """You are a patient teacher. Your responses should be:
- Clear and easy to understand
- Use analogies and real-world examples
- Break complex concepts into simple parts
- Encouraging and supportive
Focus on clarity and effective teaching.""",

    "analytical": """You are an analytical expert. Your responses should be:
- Data-driven and objective
- Include pros and cons analysis
- Present multiple perspectives
- Provide actionable recommendations
Focus on analysis and insights.""",

    "conversational": """You are a friendly assistant. Your responses should be:
- Warm and approachable
- Natural and casual in tone
- Concise but helpful
Focus on being helpful and friendly."""
}


# ============================================================================
# SOURCE DESCRIPTIONS (for Agentic Routing)
# ============================================================================
SOURCE_DESCRIPTIONS = {
    "cv_resume": "CV/Resume - education, skills, work experience, background",
    "dms_info": "DMS Research - Deep Mutational Scanning project information",
    "llm_interview": "LLM Interview - AI/ML technical concepts and interview prep"
}


# ============================================================================
# TEST CONFIG
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("CONFIG TEST - Setting up vectorstores")
    print("=" * 60)
    
    vectorstores = setup_vectorstores()
    
    print(f"\n📊 Loaded {len(vectorstores)} vectorstores:")
    for name in vectorstores.keys():
        print(f"   ✓ {name}")
    
    # Test search
    if vectorstores:
        question = "Where did Otabek study?"
        print(f"\n🔍 Test search: {question}")
        docs = search_all_vectorstores(question, vectorstores, k=2)
        print(f"   Found {len(docs)} documents")
        
        if docs:
            context = format_docs(docs)
            answer = generate_answer(question, context)
            print(f"\n💡 Answer: {answer[:300]}...")
    
    print("\n✅ Config test complete!")
