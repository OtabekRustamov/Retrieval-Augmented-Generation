"""
================================================================================
RAG ROUTING PATTERNS - STREAMLIT APP
================================================================================
Interactive demo of all 4 routing patterns:
1. Data Source Routing - WHERE to get data
2. Component Routing - HOW to process
3. Prompt Template Routing - WHAT style
4. Agentic Routing - Agent decides dynamically

Run: streamlit run app.py
================================================================================
"""

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Literal

# Import from config
from config import (
    llm, setup_vectorstores, format_docs,
    PROMPT_TEMPLATES, SOURCE_DESCRIPTIONS
)


# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="RAG Routing Patterns",
    page_icon="🔀",
    layout="wide"
)


# =============================================================================
# ROUTING SCHEMAS
# =============================================================================
class DataSourceRoute(BaseModel):
    datasource: Literal["documents", "database", "api", "general_llm"]
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)

class ComponentRoute(BaseModel):
    component: Literal["agent", "vectorstore", "llm"]
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)

class PromptRoute(BaseModel):
    prompt_type: Literal["technical", "creative", "educational", "analytical", "conversational"]
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)


# =============================================================================
# ROUTERS (CACHED)
# =============================================================================
@st.cache_resource
def create_routers():
    """Create all routers (cached for performance)."""
    
    # Data Source Router
    ds_prompt = ChatPromptTemplate.from_messages([
        ("system", """Route to data source:
- documents: SOPs, policies, PDFs, CVs
- database: Sales, metrics, records (SQL)
- api: CRM, payments, external services
- general_llm: General knowledge, definitions"""),
        ("human", "{question}")
    ])
    ds_router = ds_prompt | llm.with_structured_output(DataSourceRoute)
    
    # Component Router
    comp_prompt = ChatPromptTemplate.from_messages([
        ("system", """Route to component:
- agent: Complex multi-step, calculations, analysis
- vectorstore: Document retrieval, specific info lookup
- llm: Simple questions, definitions, general knowledge"""),
        ("human", "{question}")
    ])
    comp_router = comp_prompt | llm.with_structured_output(ComponentRoute)
    
    # Prompt Router
    prompt_prompt = ChatPromptTemplate.from_messages([
        ("system", """Route to prompt style:
- technical: Coding, engineering, APIs
- creative: Writing, stories, brainstorming
- educational: Explain, teach, tutorials
- analytical: Compare, analyze, pros/cons
- conversational: Casual chat, simple Q&A"""),
        ("human", "{question}")
    ])
    prompt_router = prompt_prompt | llm.with_structured_output(PromptRoute)
    
    return ds_router, comp_router, prompt_router


# =============================================================================
# VECTORSTORE SETUP (CACHED)
# =============================================================================
@st.cache_resource
def get_vectorstores():
    """Load vectorstores (cached)."""
    return setup_vectorstores()


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # Pattern Selection
    pattern = st.selectbox(
        "🎯 Select Routing Pattern:",
        [
            "1️⃣ Data Source Routing",
            "2️⃣ Component Routing",
            "3️⃣ Prompt Template Routing",
            "4️⃣ Agentic Routing"
        ]
    )
    
    st.markdown("---")
    
    # Pattern Info
    pattern_info = {
        "1️⃣ Data Source Routing": {
            "question": "WHERE to get data?",
            "routes": "Documents / Database / API / LLM",
            "description": "Routes to different data sources based on query type."
        },
        "2️⃣ Component Routing": {
            "question": "HOW to process?",
            "routes": "Agent / VectorStore / LLM",
            "description": "Routes to different processing components based on complexity."
        },
        "3️⃣ Prompt Template Routing": {
            "question": "WHAT style?",
            "routes": "Technical / Creative / Educational / Analytical",
            "description": "Routes to different prompt templates for different response styles."
        },
        "4️⃣ Agentic Routing": {
            "question": "Agent decides!",
            "routes": "Multiple tools dynamically",
            "description": "Agent dynamically selects and uses multiple tools."
        }
    }
    
    info = pattern_info[pattern]
    st.markdown(f"**Question:** {info['question']}")
    st.markdown(f"**Routes:** {info['routes']}")
    st.markdown(f"_{info['description']}_")
    
    st.markdown("---")
    
    # Loaded Documents
    st.markdown("### 📚 Documents")
    vectorstores = get_vectorstores()
    for name in vectorstores.keys():
        st.markdown(f"✅ {name}")


# =============================================================================
# MAIN CONTENT
# =============================================================================
st.markdown("# 🔀 RAG Routing Patterns Demo")
st.markdown("*Interactive demo of query routing techniques*")

# Get routers
ds_router, comp_router, prompt_router = create_routers()

# Query Input
question = st.text_input(
    "Enter your question:",
    placeholder="e.g., Where did Otabek study?"
)

col1, col2 = st.columns([1, 4])
with col1:
    run_button = st.button("🚀 Route & Answer", type="primary")

if run_button and question:
    with st.spinner("Processing..."):
        
        # =================================================================
        # PATTERN 1: DATA SOURCE ROUTING
        # =================================================================
        if pattern == "1️⃣ Data Source Routing":
            route = ds_router.invoke({"question": question})
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("### 🔀 Routing Decision")
                st.metric("Data Source", route.datasource.upper())
                st.metric("Confidence", f"{route.confidence:.0%}")
                st.info(f"**Reasoning:** {route.reasoning}")
            
            with col2:
                st.markdown("### 💡 Answer")
                
                if route.datasource == "documents":
                    all_docs = []
                    for vs in vectorstores.values():
                        all_docs.extend(vs.similarity_search(question, k=2))
                    
                    if all_docs:
                        context = format_docs(all_docs[:4])
                        prompt = ChatPromptTemplate.from_messages([
                            ("system", "Answer based on context:\n\n{context}"),
                            ("human", "{question}")
                        ])
                        answer = (prompt | llm | StrOutputParser()).invoke({
                            "question": question, "context": context
                        })
                        
                        with st.expander("📚 Retrieved Documents"):
                            for i, doc in enumerate(all_docs[:3]):
                                st.markdown(f"**{i+1}.** {doc.page_content[:200]}...")
                    else:
                        answer = "No relevant documents found."
                
                elif route.datasource == "database":
                    answer = "🗄️ **SQL Query Generated:**\n```sql\nSELECT * FROM data WHERE ...\n```\n\n*(In production, this would execute against your database)*"
                
                elif route.datasource == "api":
                    answer = f"🌐 **API Call Recommendation:**\nEndpoint: /search\nQuery: {question}\n\n*(In production, this would call actual APIs)*"
                
                else:
                    answer = llm.invoke(question).content
                
                st.write(answer)
        
        # =================================================================
        # PATTERN 2: COMPONENT ROUTING
        # =================================================================
        elif pattern == "2️⃣ Component Routing":
            route = comp_router.invoke({"question": question})
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("### 🔀 Routing Decision")
                st.metric("Component", route.component.upper())
                st.metric("Confidence", f"{route.confidence:.0%}")
                st.info(f"**Reasoning:** {route.reasoning}")
            
            with col2:
                st.markdown("### 💡 Answer")
                
                if route.component == "vectorstore":
                    all_docs = []
                    for vs in vectorstores.values():
                        all_docs.extend(vs.similarity_search(question, k=2))
                    
                    if all_docs:
                        context = format_docs(all_docs[:4])
                        prompt = ChatPromptTemplate.from_messages([
                            ("system", "Answer based on context:\n\n{context}"),
                            ("human", "{question}")
                        ])
                        answer = (prompt | llm | StrOutputParser()).invoke({
                            "question": question, "context": context
                        })
                        
                        with st.expander("📚 Retrieved Documents"):
                            for i, doc in enumerate(all_docs[:3]):
                                st.markdown(f"**{i+1}.** {doc.page_content[:200]}...")
                    else:
                        answer = "No relevant documents found."
                
                elif route.component == "agent":
                    st.info("🤖 Using Agent... (See Pattern 4 for full agent)")
                    answer = llm.invoke(f"Think step by step: {question}").content
                
                else:
                    answer = llm.invoke(question).content
                
                st.write(answer)
        
        # =================================================================
        # PATTERN 3: PROMPT TEMPLATE ROUTING
        # =================================================================
        elif pattern == "3️⃣ Prompt Template Routing":
            route = prompt_router.invoke({"question": question})
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("### 🔀 Routing Decision")
                st.metric("Prompt Style", route.prompt_type.upper())
                st.metric("Confidence", f"{route.confidence:.0%}")
                st.info(f"**Reasoning:** {route.reasoning}")
                
                with st.expander("📝 Selected Prompt"):
                    st.code(PROMPT_TEMPLATES[route.prompt_type][:200] + "...", language=None)
            
            with col2:
                st.markdown("### 💡 Answer")
                
                system_prompt = PROMPT_TEMPLATES.get(route.prompt_type, PROMPT_TEMPLATES["conversational"])
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{question}")
                ])
                answer = (prompt | llm | StrOutputParser()).invoke({"question": question})
                st.write(answer)
        
        # =================================================================
        # PATTERN 4: AGENTIC ROUTING
        # =================================================================
        elif pattern == "4️⃣ Agentic Routing":
            st.markdown("### 🤖 Agent Processing")
            
            # Define tools
            @tool
            def calculator(expression: str) -> str:
                """Calculate mathematical expressions."""
                try:
                    return f"Result: {eval(expression)}"
                except:
                    return "Calculation error"
            
            @tool
            def search_documents(query: str) -> str:
                """Search all documents for relevant information."""
                results = []
                for name, vs in vectorstores.items():
                    docs = vs.similarity_search(query, k=2)
                    for d in docs:
                        results.append(f"[{name}] {d.page_content[:300]}")
                return "\n\n".join(results) if results else "No results found."
            
            tools = [calculator, search_documents]
            
            agent_prompt = ChatPromptTemplate.from_messages([
                ("system", "You have tools: calculator, search_documents. Use them to answer thoroughly."),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])
            
            agent = create_openai_tools_agent(llm, tools, agent_prompt)
            executor = AgentExecutor(agent=agent, tools=tools, verbose=False, return_intermediate_steps=True)
            
            result = executor.invoke({"input": question})
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("### 🔧 Tools Used")
                tools_used = []
                for step in result.get("intermediate_steps", []):
                    action, obs = step
                    tools_used.append(action.tool)
                    st.markdown(f"**{action.tool}**")
                    st.code(str(action.tool_input)[:100], language=None)
                
                if not tools_used:
                    st.markdown("*No tools used*")
            
            with col2:
                st.markdown("### 💡 Answer")
                st.write(result["output"])


# =============================================================================
# COMPARISON TABLE
# =============================================================================
st.markdown("---")
st.markdown("### 📊 Quick Comparison")

comparison_data = """
| Pattern | Question | Routes To | Speed | Cost | Best For |
|---------|----------|-----------|-------|------|----------|
| **Data Source** | WHERE? | Docs/DB/API/LLM | ⚡ Fast | 💰 Low | Multi-source systems |
| **Component** | HOW? | Agent/VectorStore/LLM | ⚡ Fast | 💰 Low | Varying complexity |
| **Prompt** | WHAT style? | Different prompts → LLM | ⚡ Fast | 💰 Low | Style adaptation |
| **Agentic** | Agent decides! | Multiple tools | 🐢 Slow | 💰💰💰 High | Complex queries |
"""
st.markdown(comparison_data)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #888;">
        🔀 RAG Routing Patterns | Built with LangChain & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
