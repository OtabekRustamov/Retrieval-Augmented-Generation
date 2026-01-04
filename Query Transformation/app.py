"""
============================================================================
RAG QUERY TRANSFORMATION TECHNIQUES - HUGGINGFACE VERSION
============================================================================
All 6 techniques using HuggingFace models (no API key required)

Techniques:
1. Multi-Query Generation
2. RAG-Fusion (with RRF)
3. HyDE (Hypothetical Document Embeddings)
4. Query Decomposition
5. Step-Back Prompting
6. RRR (Rewrite-Retrieve-Read)
============================================================================
"""

import os
import torch
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_PATH = "D:/LLM/RAG/data/"
VECTOR_STORE_PATH = "faiss_index_hf"

# Embedding model (runs locally, no API needed)
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# LLM options (in order of preference)
# 1. Ollama (if installed)
# 2. HuggingFace local model with quantization


# ============================================================================
# MODEL INITIALIZATION
# ============================================================================
print("Initializing models...")

# Embeddings (always works locally)
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)
print(f"  Embeddings: {EMBEDDING_MODEL}")


def get_llm():
    """Get LLM - tries Ollama first, then HuggingFace"""
    
    # Try Ollama first
    try:
        llm = Ollama(model="mistral", temperature=0)
        llm.invoke("test")
        print("  LLM: Ollama (mistral)")
        return llm
    except:
        pass
    
    # Try HuggingFace with quantization
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
        
        model_id = "mistralai/Mistral-7B-Instruct-v0.2"
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto"
        )
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.1
        )
        
        llm = HuggingFacePipeline(pipeline=pipe)
        print(f"  LLM: {model_id} (4-bit quantized)")
        return llm
    except Exception as e:
        print(f"  LLM: None available - {e}")
        return None


llm = get_llm()


# ============================================================================
# VECTORSTORE SETUP
# ============================================================================
def setup_vectorstore(force_rebuild=False):
    """Load or create vectorstore"""
    
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
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(DATA_PATH, file))
            docs = loader.load()
            
            for doc in docs:
                doc.metadata["source_type"] = "pdf"
                doc.metadata["filename"] = file
            
            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)
            print(f"  {file}: {len(chunks)} chunks")
    
    print(f"\nTotal chunks: {len(all_chunks)}")
    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    vectorstore.save_local(VECTOR_STORE_PATH)
    print(f"Saved to {VECTOR_STORE_PATH}\n")
    
    return vectorstore


def format_docs(docs):
    """Format documents into context string"""
    return "\n\n".join([doc.page_content for doc in docs])


def get_retriever(vectorstore, k=5):
    """Get retriever from vectorstore"""
    return vectorstore.as_retriever(search_kwargs={"k": k})


# ============================================================================
# TECHNIQUE 1: MULTI-QUERY
# ============================================================================
def generate_multi_queries(question, num_variations=3):
    """Generate multiple query variations"""
    
    if llm is None:
        # Fallback: simple word variations
        return [question]
    
    template = """Generate {num_variations} different versions of this question.
Each version should ask the same thing with different words.

Original: {question}

Provide numbered alternatives:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({
        "question": question,
        "num_variations": num_variations
    })
    
    queries = [question]
    for line in response.strip().split("\n"):
        line = line.strip()
        if line and len(line) > 5:
            if line[0].isdigit():
                line = line.split(". ", 1)[-1].split(") ", 1)[-1]
            queries.append(line.strip())
    
    return queries[:num_variations + 1]


def multi_query_retrieve(question, retriever, num_variations=3):
    """Retrieve using multiple query variations"""
    
    queries = generate_multi_queries(question, num_variations)
    
    print("Generated Queries:")
    for i, q in enumerate(queries):
        prefix = "  (original)" if i == 0 else f"  {i}."
        print(f"{prefix} {q}")
    
    all_docs = []
    seen = set()
    
    for query in queries:
        docs = retriever.invoke(query)
        for doc in docs:
            fp = doc.page_content[:100]
            if fp not in seen:
                seen.add(fp)
                all_docs.append(doc)
    
    print(f"\nRetrieved {len(all_docs)} unique documents")
    return all_docs, queries


def multi_query_rag(question, retriever, num_variations=3):
    """Complete Multi-Query RAG pipeline"""
    
    print("\n" + "="*60)
    print("MULTI-QUERY RAG")
    print("="*60)
    
    docs, queries = multi_query_retrieve(question, retriever, num_variations)
    
    if llm is None:
        return "LLM not available"
    
    context = format_docs(docs[:5])
    
    template = """Answer based on the context. Say "Not found" if not in context.

Context:
{context}

Question: {question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    
    print(f"\nAnswer: {answer}")
    return answer


# ============================================================================
# TECHNIQUE 2: RAG-FUSION (with RRF)
# ============================================================================
def reciprocal_rank_fusion(results_list, k=60):
    """Combine results using RRF scoring"""
    
    fused_scores = {}
    doc_map = {}
    
    for docs in results_list:
        for rank, doc in enumerate(docs):
            doc_id = doc.page_content[:100]
            
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
                doc_map[doc_id] = doc
            
            fused_scores[doc_id] += 1 / (k + rank)
    
    sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_map[doc_id], score) for doc_id, score in sorted_docs]


def rag_fusion_retrieve(question, retriever, num_variations=3):
    """RAG-Fusion retrieval with RRF"""
    
    queries = generate_multi_queries(question, num_variations)
    
    print("Generated Queries:")
    for i, q in enumerate(queries):
        prefix = "  (original)" if i == 0 else f"  {i}."
        print(f"{prefix} {q}")
    
    # Retrieve for each query
    results_list = [retriever.invoke(q) for q in queries]
    
    # Apply RRF
    ranked = reciprocal_rank_fusion(results_list)
    
    print(f"\nRRF Ranked Results (top 5):")
    for i, (doc, score) in enumerate(ranked[:5]):
        source = doc.metadata.get("filename", "unknown")
        print(f"  {i+1}. Score: {score:.4f} | {source}")
    
    return ranked, queries


def rag_fusion_rag(question, retriever, num_variations=3):
    """Complete RAG-Fusion pipeline"""
    
    print("\n" + "="*60)
    print("RAG-FUSION")
    print("="*60)
    
    ranked, queries = rag_fusion_retrieve(question, retriever, num_variations)
    
    if llm is None:
        return "LLM not available"
    
    docs = [doc for doc, _ in ranked[:5]]
    context = format_docs(docs)
    
    template = """Answer based on the context. Say "Not found" if not in context.

Context:
{context}

Question: {question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    
    print(f"\nAnswer: {answer}")
    return answer


# ============================================================================
# TECHNIQUE 3: HyDE
# ============================================================================
def generate_hypothetical(question):
    """Generate hypothetical answer"""
    
    if llm is None:
        return question
    
    template = """Write a short paragraph that would answer this question.
Even if you don't know the real answer, write what a good answer might look like.

Question: {question}

Hypothetical answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question})


def hyde_retrieve(question, vectorstore, k=5):
    """HyDE retrieval using hypothetical document"""
    
    print(f"Original Question: {question}\n")
    
    hypothetical = generate_hypothetical(question)
    print(f"Hypothetical Answer:\n{hypothetical[:200]}...\n")
    
    # Embed hypothetical and search
    docs = vectorstore.similarity_search(hypothetical, k=k)
    
    print(f"Retrieved {len(docs)} documents")
    return docs, hypothetical


def hyde_rag(question, vectorstore, k=5):
    """Complete HyDE RAG pipeline"""
    
    print("\n" + "="*60)
    print("HyDE (Hypothetical Document Embeddings)")
    print("="*60)
    
    docs, hypothetical = hyde_retrieve(question, vectorstore, k)
    
    if llm is None:
        return "LLM not available"
    
    context = format_docs(docs)
    
    template = """Answer based on the context. Say "Not found" if not in context.

Context:
{context}

Question: {question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    
    print(f"\nAnswer: {answer}")
    return answer


# ============================================================================
# TECHNIQUE 4: QUERY DECOMPOSITION
# ============================================================================
def decompose_query(question, max_sub=4):
    """Break complex question into sub-questions"""
    
    if llm is None:
        return [question]
    
    template = """Break this complex question into {max_sub} simple sub-questions.
Each should ask about ONE specific aspect.

Complex question: {question}

Sub-questions (numbered):"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({"question": question, "max_sub": max_sub})
    
    sub_questions = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if line and len(line) > 5:
            if line[0].isdigit():
                line = line.split(". ", 1)[-1].split(") ", 1)[-1]
            sub_questions.append(line.strip())
    
    return sub_questions[:max_sub]


def decomposition_rag(question, retriever):
    """Complete Query Decomposition RAG pipeline"""
    
    print("\n" + "="*60)
    print("QUERY DECOMPOSITION")
    print("="*60)
    print(f"Complex Question: {question}\n")
    
    sub_questions = decompose_query(question)
    
    print("Sub-questions:")
    for i, sq in enumerate(sub_questions):
        print(f"  {i+1}. {sq}")
    
    if llm is None:
        return "LLM not available"
    
    # Answer each sub-question
    sub_answers = []
    for sq in sub_questions:
        docs = retriever.invoke(sq)
        context = format_docs(docs[:3])
        
        template = """Answer concisely based on context.

Context: {context}

Question: {question}

Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": sq})
        sub_answers.append({"question": sq, "answer": answer})
    
    # Synthesize final answer
    qa_pairs = "\n".join([f"Q: {sa['question']}\nA: {sa['answer']}" for sa in sub_answers])
    
    template = """Based on these Q&A pairs, provide a comprehensive answer.

{qa_pairs}

Original question: {question}

Synthesized answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    final_answer = chain.invoke({"qa_pairs": qa_pairs, "question": question})
    
    print(f"\nFinal Answer: {final_answer}")
    return final_answer


# ============================================================================
# TECHNIQUE 5: STEP-BACK PROMPTING
# ============================================================================
def generate_step_back(question):
    """Generate broader step-back question"""
    
    if llm is None:
        return question
    
    template = """Generate a broader, more general version of this question.
The step-back question should ask about principles or background knowledge.

Specific question: {question}

General step-back question:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question}).strip()


def step_back_rag(question, retriever):
    """Complete Step-Back RAG pipeline"""
    
    print("\n" + "="*60)
    print("STEP-BACK PROMPTING")
    print("="*60)
    print(f"Specific Question: {question}\n")
    
    step_back_q = generate_step_back(question)
    print(f"Step-Back Question: {step_back_q}\n")
    
    # Retrieve for both
    general_docs = retriever.invoke(step_back_q)
    specific_docs = retriever.invoke(question)
    
    print(f"Retrieved: {len(general_docs)} general + {len(specific_docs)} specific docs")
    
    if llm is None:
        return "LLM not available"
    
    general_context = format_docs(general_docs[:3])
    specific_context = format_docs(specific_docs[:3])
    
    template = """Answer using both general principles and specific details.

GENERAL CONTEXT (principles):
{general_context}

SPECIFIC CONTEXT (details):
{specific_context}

Question: {question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "general_context": general_context,
        "specific_context": specific_context,
        "question": question
    })
    
    print(f"\nAnswer: {answer}")
    return answer


# ============================================================================
# TECHNIQUE 6: RRR (REWRITE-RETRIEVE-READ)
# ============================================================================
def rewrite_query(question):
    """Rewrite vague query to clear one"""
    
    if llm is None:
        return question
    
    template = """Rewrite this vague or messy question into a clear, searchable query.
Use specific technical terms.

Original: {question}

Rewritten:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question}).strip()


def rrr_rag(question, retriever):
    """Complete RRR (Rewrite-Retrieve-Read) pipeline"""
    
    print("\n" + "="*60)
    print("RRR (REWRITE-RETRIEVE-READ)")
    print("="*60)
    print(f"Original Query: {question}\n")
    
    rewritten = rewrite_query(question)
    print(f"Rewritten Query: {rewritten}\n")
    
    # Retrieve with rewritten query
    docs = retriever.invoke(rewritten)
    print(f"Retrieved {len(docs)} documents")
    
    if llm is None:
        return "LLM not available"
    
    context = format_docs(docs[:5])
    
    # Answer original question
    template = """Answer the ORIGINAL question based on context.

Context:
{context}

Original Question: {question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    
    print(f"\nAnswer: {answer}")
    return answer


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    
    print("\n" + "="*60)
    print("QUERY TRANSFORMATION TECHNIQUES - HUGGINGFACE")
    print("="*60 + "\n")
    
    # Setup
    vectorstore = setup_vectorstore()
    retriever = get_retriever(vectorstore, k=5)
    
    # Test questions
    test_questions = [
        "Where did Otabek study?",
        "What is DMS?",
        "Compare the different molecular tools",
    ]
    
    # Test each technique
    print("\n" + "="*60)
    print("TEST 1: MULTI-QUERY")
    print("="*60)
    multi_query_rag(test_questions[0], retriever)
    
    print("\n" + "="*60)
    print("TEST 2: RAG-FUSION")
    print("="*60)
    rag_fusion_rag(test_questions[0], retriever)
    
    print("\n" + "="*60)
    print("TEST 3: HyDE")
    print("="*60)
    hyde_rag(test_questions[1], vectorstore)
    
    print("\n" + "="*60)
    print("TEST 4: DECOMPOSITION")
    print("="*60)
    decomposition_rag(test_questions[2], retriever)
    
    print("\n" + "="*60)
    print("TEST 5: STEP-BACK")
    print("="*60)
    step_back_rag(test_questions[1], retriever)
    
    print("\n" + "="*60)
    print("TEST 6: RRR")
    print("="*60)
    rrr_rag("that molecule stuff", retriever)
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)
