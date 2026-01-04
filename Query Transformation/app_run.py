"""
============================================================================
RAG QUERY TRANSFORMATION - STREAMLIT APP (OpenAI)
============================================================================
"""

import streamlit as st
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# API Key
from sk import my_gpt as API_KEY


# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_PATH = "D:/LLM/RAG/data/"
VECTOR_STORE_PATH = "faiss_index"


# ============================================================================
# MODELS
# ============================================================================
@st.cache_resource
def get_models():
    embeddings = OpenAIEmbeddings(api_key=API_KEY, model="text-embedding-3-small")
    llm = ChatOpenAI(api_key=API_KEY, model="gpt-4o-mini", temperature=0)
    return embeddings, llm


# ============================================================================
# VECTORSTORE
# ============================================================================
@st.cache_resource
def setup_vectorstore():
    embeddings, _ = get_models()
    
    if os.path.exists(VECTOR_STORE_PATH):
        return FAISS.load_local(
            VECTOR_STORE_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(DATA_PATH, file))
            docs = loader.load()
            for doc in docs:
                doc.metadata["filename"] = file
            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)
    
    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    vectorstore.save_local(VECTOR_STORE_PATH)
    return vectorstore


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def parse_numbered_list(response):
    items = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if line and len(line) > 5:
            if line[0].isdigit():
                line = line.split(". ", 1)[-1].split(") ", 1)[-1]
            items.append(line.strip())
    return items


# ============================================================================
# TECHNIQUE IMPLEMENTATIONS
# ============================================================================
def multi_query(question, vectorstore, llm, k=5):
    """Multi-Query technique"""
    
    # Generate variations
    template = """Generate 3 different versions of this question.
Each should ask the same thing with different words.

Original: {question}

Alternatives (numbered):"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"question": question})
    
    queries = [question] + parse_numbered_list(response)[:3]
    
    # Retrieve for each query
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    all_docs = []
    seen = set()
    
    for q in queries:
        docs = retriever.invoke(q)
        for doc in docs:
            fp = doc.page_content[:100]
            if fp not in seen:
                seen.add(fp)
                all_docs.append(doc)
    
    # Generate answer
    context = format_docs(all_docs[:k])
    
    answer_template = """Answer based on context. Say "Not found" if not available.

Context: {context}

Question: {question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(answer_template)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    
    return {
        "answer": answer,
        "queries": queries,
        "documents": all_docs[:k]
    }


def rag_fusion(question, vectorstore, llm, k=5):
    """RAG-Fusion with RRF"""
    
    # Generate variations
    template = """Generate 3 different versions of this question.

Original: {question}

Alternatives:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"question": question})
    
    queries = [question] + parse_numbered_list(response)[:3]
    
    # Retrieve and apply RRF
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    results_list = [retriever.invoke(q) for q in queries]
    
    # RRF scoring
    fused_scores = {}
    doc_map = {}
    
    for docs in results_list:
        for rank, doc in enumerate(docs):
            doc_id = doc.page_content[:100]
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
                doc_map[doc_id] = doc
            fused_scores[doc_id] += 1 / (60 + rank)
    
    sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ranked_docs = [(doc_map[doc_id], score) for doc_id, score in sorted_docs]
    
    # Generate answer
    docs = [doc for doc, _ in ranked_docs[:k]]
    context = format_docs(docs)
    
    answer_template = """Answer based on context.

Context: {context}

Question: {question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(answer_template)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    
    return {
        "answer": answer,
        "queries": queries,
        "ranked_docs": ranked_docs[:k],
        "documents": docs
    }


def hyde(question, vectorstore, llm, k=5):
    """HyDE technique"""
    
    # Generate hypothetical
    template = """Write a short paragraph answering this question.
Even if unsure, write what a good answer might look like.

Question: {question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    hypothetical = chain.invoke({"question": question})
    
    # Retrieve using hypothetical
    docs = vectorstore.similarity_search(hypothetical, k=k)
    context = format_docs(docs)
    
    # Generate final answer
    answer_template = """Answer based on context.

Context: {context}

Question: {question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(answer_template)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    
    return {
        "answer": answer,
        "hypothetical": hypothetical,
        "documents": docs
    }


def decomposition(question, vectorstore, llm, k=5):
    """Query Decomposition"""
    
    # Decompose
    template = """Break this into 3-4 simple sub-questions.

Complex: {question}

Sub-questions:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"question": question})
    
    sub_questions = parse_numbered_list(response)[:4]
    
    # Answer each
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    sub_answers = []
    
    for sq in sub_questions:
        docs = retriever.invoke(sq)
        context = format_docs(docs)
        
        template = """Answer concisely.

Context: {context}

Question: {question}

Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": sq})
        sub_answers.append({"question": sq, "answer": answer})
    
    # Synthesize
    qa_pairs = "\n".join([f"Q: {sa['question']}\nA: {sa['answer']}" for sa in sub_answers])
    
    template = """Synthesize a comprehensive answer.

{qa_pairs}

Original: {question}

Final answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    final_answer = chain.invoke({"qa_pairs": qa_pairs, "question": question})
    
    return {
        "answer": final_answer,
        "sub_questions": sub_answers,
        "documents": []
    }


def step_back(question, vectorstore, llm, k=5):
    """Step-Back Prompting"""
    
    # Generate step-back question
    template = """Generate a broader version of this question.
Ask about general principles or background.

Specific: {question}

General:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    step_back_q = chain.invoke({"question": question}).strip()
    
    # Retrieve for both
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    general_docs = retriever.invoke(step_back_q)
    specific_docs = retriever.invoke(question)
    
    # Generate answer
    general_context = format_docs(general_docs)
    specific_context = format_docs(specific_docs)
    
    template = """Answer using both contexts.

GENERAL (principles):
{general_context}

SPECIFIC (details):
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
    
    all_docs = list({doc.page_content[:100]: doc for doc in general_docs + specific_docs}.values())
    
    return {
        "answer": answer,
        "step_back_question": step_back_q,
        "general_count": len(general_docs),
        "specific_count": len(specific_docs),
        "documents": all_docs
    }


def rrr(question, vectorstore, llm, k=5):
    """RRR (Rewrite-Retrieve-Read)"""
    
    # Rewrite
    template = """Rewrite this into a clear, searchable query.

Original: {question}

Rewritten:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    rewritten = chain.invoke({"question": question}).strip()
    
    # Retrieve with rewritten
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(rewritten)
    context = format_docs(docs)
    
    # Answer original
    template = """Answer the ORIGINAL question.

Context: {context}

Original: {question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    
    return {
        "answer": answer,
        "rewritten": rewritten,
        "documents": docs
    }


# ============================================================================
# STREAMLIT APP
# ============================================================================
def main():
    st.set_page_config(
        page_title="Query Transformation", 
        page_icon="Q", 
        layout="wide"
    )
    
    st.title("RAG Query Transformation Techniques")
    st.markdown("*Transform queries for better retrieval*")
    
    # Initialize
    with st.spinner("Loading..."):
        embeddings, llm = get_models()
        vectorstore = setup_vectorstore()
    
    # Sidebar
    with st.sidebar:
        st.header("Select Technique")
        
        technique = st.radio(
            "Technique:",
            options=[
                ("multi_query", "Multi-Query"),
                ("rag_fusion", "RAG-Fusion"),
                ("hyde", "HyDE"),
                ("decomposition", "Decomposition"),
                ("step_back", "Step-Back"),
                ("rrr", "RRR")
            ],
            format_func=lambda x: x[1]
        )[0]
        
        st.markdown("---")
        
        descriptions = {
            "multi_query": "Generate multiple query variations",
            "rag_fusion": "Multi-Query + RRF ranking",
            "hyde": "Generate hypothetical answer first",
            "decomposition": "Break into sub-questions",
            "step_back": "Generate broader question",
            "rrr": "Rewrite vague query"
        }
        
        st.markdown(f"**{technique.replace('_', ' ').title()}**")
        st.markdown(f"_{descriptions[technique]}_")
    
    # Main area
    question = st.text_input(
        "Enter your question:",
        placeholder="e.g., Where did Otabek study?"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        search_btn = st.button("Search", type="primary")
    
    if search_btn and question:
        with st.spinner(f"Processing with {technique}..."):
            
            # Call appropriate technique
            if technique == "multi_query":
                result = multi_query(question, vectorstore, llm)
            elif technique == "rag_fusion":
                result = rag_fusion(question, vectorstore, llm)
            elif technique == "hyde":
                result = hyde(question, vectorstore, llm)
            elif technique == "decomposition":
                result = decomposition(question, vectorstore, llm)
            elif technique == "step_back":
                result = step_back(question, vectorstore, llm)
            else:
                result = rrr(question, vectorstore, llm)
        
        # Display answer
        st.subheader("Answer")
        st.write(result["answer"])
        
        # Display technique-specific info
        st.subheader("Process Details")
        
        if technique == "multi_query":
            st.markdown("**Generated Queries:**")
            for i, q in enumerate(result.get("queries", [])):
                marker = "(original)" if i == 0 else f"{i}."
                st.markdown(f"- {marker} {q}")
        
        elif technique == "rag_fusion":
            st.markdown("**Generated Queries:**")
            for i, q in enumerate(result.get("queries", [])):
                st.markdown(f"- {q}")
            
            st.markdown("**RRF Scores:**")
            for i, (doc, score) in enumerate(result.get("ranked_docs", [])[:5]):
                st.markdown(f"- Doc {i+1}: {score:.4f}")
        
        elif technique == "hyde":
            st.markdown("**Hypothetical Answer:**")
            st.text_area("", result.get("hypothetical", ""), height=100, disabled=True)
        
        elif technique == "decomposition":
            st.markdown("**Sub-questions:**")
            for item in result.get("sub_questions", []):
                with st.expander(f"Q: {item['question']}"):
                    st.write(item['answer'])
        
        elif technique == "step_back":
            st.markdown(f"**Step-Back Question:** {result.get('step_back_question', '')}")
            st.markdown(f"General: {result.get('general_count', 0)} docs, Specific: {result.get('specific_count', 0)} docs")
        
        elif technique == "rrr":
            st.markdown(f"**Original:** {question}")
            st.markdown(f"**Rewritten:** {result.get('rewritten', '')}")
        
        # Display documents
        if result.get("documents"):
            st.subheader("Retrieved Documents")
            for i, doc in enumerate(result["documents"][:5]):
                source = doc.metadata.get("filename", "unknown")
                with st.expander(f"Document {i+1} - {source}"):
                    st.write(doc.page_content[:500] + "...")
    
    elif search_btn:
        st.warning("Please enter a question.")


if __name__ == "__main__":
    main()
