# import streamlit as st
# import os
# from datetime import datetime
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import (
#     PyPDFLoader, UnstructuredPDFLoader, UnstructuredImageLoader
# )
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.prompts import PromptTemplate
# from langchain_core.messages import HumanMessage, AIMessage
# from langgraph.graph import StateGraph, END, START
# from typing import Sequence
# from typing_extensions import TypedDict
# import tempfile

# # ---- Set your API key and config ----
# # os.environ["GOOGLE_API_KEY"] ="AIzaSyCKlOwIgwC6IQXmXS9eVOt4iaHkUGadDPA"
# from dotenv import load_dotenv
# load_dotenv()
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# VECTOR_DB_PATH = "vectorstore"
# EMBED_MODEL = "models/embedding-001"

# st.set_page_config(page_title="Legal Aid Navigator", page_icon="🧑‍⚖️")
# st.title("🇮🇳 Legal Aid Navigator — Indian Law Agentic RAG Chatbot")

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# def get_legal_prompt():
#     prompt_path = "prompts/legal_aid_prompt.txt"
#     if os.path.exists(prompt_path):
#         with open(prompt_path) as f:
#             return f.read()
#     return (
#         "You are Legal Aid Navigator, an AI paralegal built for Indian citizens. "
#         "Answer legal questions with clear citations to statutes/sections from Indian law. If unsure, say so."
#     )
# legal_prompt = get_legal_prompt()

# def is_faiss_index_present(folder):
#     return (
#         os.path.isdir(folder)
#         and os.path.exists(os.path.join(folder, "index.faiss"))
#         and os.path.exists(os.path.join(folder, "index.pkl"))
#     )

# # Safe vectorstore loading for LangChain >=0.1.16, FAISS >=1.8.0
# def load_vector_db():
#     if is_faiss_index_present(VECTOR_DB_PATH):
#         return FAISS.load_local(
#             VECTOR_DB_PATH,
#             GoogleGenerativeAIEmbeddings(model=EMBED_MODEL),
#             allow_dangerous_deserialization=True  # REQUIRED in new releases!
#         )
#     else:
#         return None

# vector_db = load_vector_db()
# retriever = vector_db.as_retriever(search_kwargs={"k":7}) if vector_db else None

# def ingest_file(file, user_id="uploaded"):
#     global vector_db, retriever
#     file_ext = file.name.split(".")[-1].lower()
#     tf = tempfile.mktemp(suffix="."+file_ext)
#     with open(tf, "wb") as fout:
#         fout.write(file.read())
#     try:
#         if file_ext == "pdf":
#             try:
#                 loader = PyPDFLoader(tf)
#                 docs = loader.load()
#             except Exception:
#                 loader = UnstructuredPDFLoader(tf)
#                 docs = loader.load()
#         elif file_ext in {"jpg", "jpeg", "png"}:
#             loader = UnstructuredImageLoader(tf)
#             docs = loader.load()
#         else:
#             st.warning(f"Unsupported file type: {file.name}")
#             return []
#     except Exception as e:
#         st.error(f"Failed to process file {file.name}: {e}")
#         return []
#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     doc_chunks = splitter.split_documents(docs)
#     for chunk in doc_chunks:
#         chunk.metadata["uploaded_by"] = user_id
#         chunk.metadata["source"] = file.name
#     if doc_chunks:
#         if vector_db is None:
#             vector_db = FAISS.from_documents(doc_chunks, GoogleGenerativeAIEmbeddings(model=EMBED_MODEL))
#             vector_db.save_local(VECTOR_DB_PATH)
#             st.success(f"✅ Knowledge base initialized with {file.name} ({len(doc_chunks)} sections)!")
#         else:
#             vector_db.add_documents(doc_chunks)
#             vector_db.save_local(VECTOR_DB_PATH)
#             st.success(f"✅ Uploaded and indexed {file.name} ({len(doc_chunks)} sections). Ready for QA!")
#         retriever = vector_db.as_retriever(search_kwargs={"k":7})
#     return doc_chunks

# # ==== Agentic RAG Workflow Section (LangGraph) ====
# class AgentState(TypedDict):
#     messages: Sequence

# def agent_node(state):
#     messages = state["messages"]
#     query = messages[-1].content if messages else ""
#     if not query or len(query) < 5:
#         ai_msg = AIMessage(content="Please clarify your question for better legal assistance.")
#         return {"messages": messages + [ai_msg]}
#     ai_msg = AIMessage(content="[[RETRIEVE]]")
#     return {"messages": messages + [ai_msg]}

# def retrieve_node(state):
#     messages = state["messages"]
#     query = None
#     for msg in reversed(messages):
#         if isinstance(msg, HumanMessage):
#             query = msg.content
#             break
#     if not retriever:
#         ai_msg = AIMessage(content="No documents are indexed yet. Please upload legal PDFs or images.")
#         return {"messages": messages + [ai_msg]}
#     docs = retriever.get_relevant_documents(query)
#     context = "\n\n".join(d.page_content for d in docs)
#     ai_msg = AIMessage(content="[CONTEXT]\n" + context,
#                        additional_kwargs={"sources": [(d.metadata.get("source", "unknown"), d.page_content[:200]) for d in docs]})
#     return {"messages": messages + [ai_msg]}

# def generate_node(state):
#     messages = state["messages"]
#     query = None
#     context = ""
#     sources = []
#     for msg in messages:
#         if isinstance(msg, HumanMessage):
#             query = msg.content
#         if isinstance(msg, AIMessage) and msg.content.startswith("[CONTEXT]"):
#             context = msg.content[len("[CONTEXT]\n") :]
#             sources = msg.additional_kwargs.get("sources", [])
#     prompt = PromptTemplate(
#         template=legal_prompt,
#         input_variables=["context", "question"]
#     ).format(context=context, question=query)
#     llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
#     answer = llm.invoke([HumanMessage(content=prompt)])
#     ai_msg = AIMessage(content=answer.content, additional_kwargs={"sources": sources})
#     return {"messages": messages + [ai_msg]}

# graph = StateGraph(AgentState)
# graph.add_node("agent", agent_node)
# graph.add_node("retrieve", retrieve_node)
# graph.add_node("generate", generate_node)
# graph.add_edge(START, "agent")
# graph.add_conditional_edges(
#     "agent",
#     lambda state: "retrieve" if "[[RETRIEVE]]" in state["messages"][-1].content else END,
#     {"retrieve": "retrieve", END: END}
# )
# graph.add_edge("retrieve", "generate")
# graph.add_edge("generate", END)
# compiled_graph = graph.compile()

# # ==== Streamlit UI ====
# st.header("📁 Add Legal PDF or Image (Real-time Indexing)")
# uploaded = st.file_uploader(
#     "Upload PDF, JPG, PNG", type=["pdf", "jpg", "jpeg", "png"], accept_multiple_files=True
# )
# user_id = st.text_input("Your Name or Identifier:", value="anon", key="user_id")
# if uploaded:
#     for file in uploaded:
#         new_chunks = ingest_file(file, user_id=user_id)
#         if new_chunks:
#             st.success(f"Indexed {len(new_chunks)} sections from '{file.name}'.")
#         else:
#             st.error(f"No readable content extracted in '{file.name}'.")

# st.markdown("---")
# st.markdown("## 💬 Legal Question Agentic Chatbot")
# for turn in st.session_state.messages:
#     if turn["role"] == "user":
#         st.markdown(f"**You:** {turn['content']}")
#     else:
#         st.markdown(f"🧑‍⚖️ **Paralegal:** {turn['content']}")
#         if turn.get("sources"):
#             with st.expander("Show sources / sections"):
#                 for metad, chunk_txt in turn["sources"]:
#                     st.write(f"**{metad}**")
#                     st.caption(chunk_txt[:350])

# with st.form("chat-form", clear_on_submit=True):
#     query = st.text_area("Your question:", placeholder="Ask about any Indian law, e.g. 'What is Section 420 IPC?'")
#     submit = st.form_submit_button("Send")

# if submit and query.strip():
#     st.session_state.messages.append({"role": "user", "content": query, "time": str(datetime.now())})
#     chat_msgs = [
#         HumanMessage(content=msg["content"]) if msg["role"] == "user"
#         else AIMessage(content=msg["content"], additional_kwargs={"sources": msg.get("sources",[])})
#         for msg in st.session_state.messages if msg["role"] in ["user", "bot"]
#     ]
#     state = {"messages": chat_msgs}
#     result = compiled_graph.invoke(state)
#     # Only add new bot turns to chat!
#     new_ai_turns = [m for m in result["messages"] if isinstance(m, AIMessage)]
#     for m in new_ai_turns:
#         st.session_state.messages.append({
#             "role": "bot",
#             "content": m.content,
#             "sources": m.additional_kwargs.get("sources", []),
#             "time": str(datetime.now())
#         })
#     st.experimental_rerun()

# st.info("*I am an AI paralegal, not a lawyer. For formal advice, consult a licensed advocate.*")






import streamlit as st
import os
import asyncio  # ✅ Fix: Required for event loop issues in Streamlit
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader, UnstructuredPDFLoader, UnstructuredImageLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, START
from typing import Sequence
from typing_extensions import TypedDict
import tempfile
from dotenv import load_dotenv

# ✅ Fix: Create an event loop to avoid RuntimeError in Streamlit threads
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# ✅ Load Google API Key securely
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

VECTOR_DB_PATH = "vectorstore"
EMBED_MODEL = "models/embedding-001"

st.set_page_config(page_title="Legal Aid Navigator", page_icon="🧑‍⚖️")
st.title("🇮🇳 Legal Aid Navigator — Indian Law Agentic RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

def get_legal_prompt():
    prompt_path = "prompts/legal_aid_prompt.txt"
    if os.path.exists(prompt_path):
        with open(prompt_path) as f:
            return f.read()
    return (
        "You are Legal Aid Navigator, an AI paralegal built for Indian citizens. "
        "Answer legal questions with clear citations to statutes/sections from Indian law. If unsure, say so."
    )
legal_prompt = get_legal_prompt()

def is_faiss_index_present(folder):
    return (
        os.path.isdir(folder)
        and os.path.exists(os.path.join(folder, "index.faiss"))
        and os.path.exists(os.path.join(folder, "index.pkl"))
    )

def load_vector_db():
    if is_faiss_index_present(VECTOR_DB_PATH):
        return FAISS.load_local(
            VECTOR_DB_PATH,
            GoogleGenerativeAIEmbeddings(model=EMBED_MODEL),
            allow_dangerous_deserialization=True
        )
    else:
        return None

vector_db = load_vector_db()
retriever = vector_db.as_retriever(search_kwargs={"k": 7}) if vector_db else None

def ingest_file(file, user_id="uploaded"):
    global vector_db, retriever
    file_ext = file.name.split(".")[-1].lower()
    tf = tempfile.mktemp(suffix="." + file_ext)
    with open(tf, "wb") as fout:
        fout.write(file.read())
    try:
        if file_ext == "pdf":
            try:
                loader = PyPDFLoader(tf)
                docs = loader.load()
            except Exception:
                loader = UnstructuredPDFLoader(tf)
                docs = loader.load()
        elif file_ext in {"jpg", "jpeg", "png"}:
            loader = UnstructuredImageLoader(tf)
            docs = loader.load()
        else:
            st.warning(f"Unsupported file type: {file.name}")
            return []
    except Exception as e:
        st.error(f"Failed to process file {file.name}: {e}")
        return []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    doc_chunks = splitter.split_documents(docs)
    for chunk in doc_chunks:
        chunk.metadata["uploaded_by"] = user_id
        chunk.metadata["source"] = file.name
    if doc_chunks:
        if vector_db is None:
            vector_db = FAISS.from_documents(doc_chunks, GoogleGenerativeAIEmbeddings(model=EMBED_MODEL))
            vector_db.save_local(VECTOR_DB_PATH)
            st.success(f"✅ Knowledge base initialized with {file.name} ({len(doc_chunks)} sections)!")
        else:
            vector_db.add_documents(doc_chunks)
            vector_db.save_local(VECTOR_DB_PATH)
            st.success(f"✅ Uploaded and indexed {file.name} ({len(doc_chunks)} sections). Ready for QA!")
        retriever = vector_db.as_retriever(search_kwargs={"k": 7})
    return doc_chunks

# ==== LangGraph Agentic RAG Setup ====

class AgentState(TypedDict):
    messages: Sequence

def agent_node(state):
    messages = state["messages"]
    query = messages[-1].content if messages else ""
    if not query or len(query) < 5:
        ai_msg = AIMessage(content="Please clarify your question for better legal assistance.")
        return {"messages": messages + [ai_msg]}
    ai_msg = AIMessage(content="[[RETRIEVE]]")
    return {"messages": messages + [ai_msg]}

def retrieve_node(state):
    messages = state["messages"]
    query = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            query = msg.content
            break
    if not retriever:
        ai_msg = AIMessage(content="No documents are indexed yet. Please upload legal PDFs or images.")
        return {"messages": messages + [ai_msg]}
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join(d.page_content for d in docs)
    ai_msg = AIMessage(
        content="[CONTEXT]\n" + context,
        additional_kwargs={"sources": [(d.metadata.get("source", "unknown"), d.page_content[:200]) for d in docs]}
    )
    return {"messages": messages + [ai_msg]}

def generate_node(state):
    messages = state["messages"]
    query = None
    context = ""
    sources = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            query = msg.content
        if isinstance(msg, AIMessage) and msg.content.startswith("[CONTEXT]"):
            context = msg.content[len("[CONTEXT]\n"):]
            sources = msg.additional_kwargs.get("sources", [])
    prompt = PromptTemplate(
        template=legal_prompt,
        input_variables=["context", "question"]
    ).format(context=context, question=query)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    answer = llm.invoke([HumanMessage(content=prompt)])
    ai_msg = AIMessage(content=answer.content, additional_kwargs={"sources": sources})
    return {"messages": messages + [ai_msg]}

# Build LangGraph
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)
graph.add_edge(START, "agent")
graph.add_conditional_edges(
    "agent",
    lambda state: "retrieve" if "[[RETRIEVE]]" in state["messages"][-1].content else END,
    {"retrieve": "retrieve", END: END}
)
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)
compiled_graph = graph.compile()

# ==== Streamlit UI ====

st.header("📁 Add Legal PDF or Image (Real-time Indexing)")
uploaded = st.file_uploader("Upload PDF, JPG, PNG", type=["pdf", "jpg", "jpeg", "png"], accept_multiple_files=True)
user_id = st.text_input("Your Name or Identifier:", value="anon", key="user_id")
if uploaded:
    for file in uploaded:
        new_chunks = ingest_file(file, user_id=user_id)
        if new_chunks:
            st.success(f"Indexed {len(new_chunks)} sections from '{file.name}'.")
        else:
            st.error(f"No readable content extracted in '{file.name}'.")

st.markdown("---")
st.markdown("## 💬 Legal Question Agentic Chatbot")
for turn in st.session_state.messages:
    if turn["role"] == "user":
        st.markdown(f"**You:** {turn['content']}")
    else:
        st.markdown(f"🧑‍⚖️ **Paralegal:** {turn['content']}")
        if turn.get("sources"):
            with st.expander("Show sources / sections"):
                for metad, chunk_txt in turn["sources"]:
                    st.write(f"**{metad}**")
                    st.caption(chunk_txt[:350])

with st.form("chat-form", clear_on_submit=True):
    query = st.text_area("Your question:", placeholder="Ask about any Indian law, e.g. 'What is Section 420 IPC?'")
    submit = st.form_submit_button("Send")

if submit and query.strip():
    st.session_state.messages.append({"role": "user", "content": query, "time": str(datetime.now())})
    chat_msgs = [
        HumanMessage(content=msg["content"]) if msg["role"] == "user"
        else AIMessage(content=msg["content"], additional_kwargs={"sources": msg.get("sources", [])})
        for msg in st.session_state.messages if msg["role"] in ["user", "bot"]
    ]
    state = {"messages": chat_msgs}
    result = compiled_graph.invoke(state)
    new_ai_turns = [m for m in result["messages"] if isinstance(m, AIMessage)]
    for m in new_ai_turns:
        st.session_state.messages.append({
            "role": "bot",
            "content": m.content,
            "sources": m.additional_kwargs.get("sources", []),
            "time": str(datetime.now())
        })
    st.rerun()  # ✅ New: replace deprecated st.experimental_rerun()

st.info("*I am an AI paralegal, not a lawyer. For formal advice, consult a licensed advocate.*")
