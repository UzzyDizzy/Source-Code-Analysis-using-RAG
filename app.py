#app.py
import os
import streamlit as st
from langchain_chroma.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from src.helper import repo_ingestion, load_embedding
from store_index import store_vectors

# -------------------------------
# 🔑 API KEY MANAGEMENT SECTION
# -------------------------------
# Clear API key on refresh (default)
if "OPENAI_API_KEY" not in st.session_state:
    st.session_state["OPENAI_API_KEY"] = None

# 1. Try secrets.toml first (optional)
default_key = None
try:
    #default_key = os.environ.get("OPENAI_API_KEY")
    default_key = st.secrets["openai"]["api_key"]
except Exception:
    pass

# 2. Sidebar key input (optional override)
st.sidebar.title("⚙️ Configurations")
entered_key = st.sidebar.text_input(
    "Enter your OpenAI API key (overrides secrets.toml):", 
    type="password", 
    placeholder="sk-proj-..."
)

# If user entered a key, use it. Else fall back to secrets.
if entered_key.strip():
    st.session_state["OPENAI_API_KEY"] = entered_key.strip()
elif default_key:
    st.session_state["OPENAI_API_KEY"] = default_key
else:
    st.session_state["OPENAI_API_KEY"] = None

# Sync key to environment for langchain
if st.session_state["OPENAI_API_KEY"]:
    os.environ["OPENAI_API_KEY"] = st.session_state["OPENAI_API_KEY"]
else:
    st.sidebar.warning("⚠️ No API key provided. Please enter one above to continue.")
    os.environ.pop("OPENAI_API_KEY", None)

# Button to clear key manually (optional)
if st.sidebar.button("🔁 Clear API Key"):
    st.session_state["OPENAI_API_KEY"] = None
    os.environ.pop("OPENAI_API_KEY", None)
    st.sidebar.success("✅ API key cleared. Enter a new one above.")


# -------------------------------
# 🧠 REPO INGESTION
# -------------------------------
repo_url = st.sidebar.text_input("Enter GitHub Repo URL:")

if st.sidebar.button("Ingest Repo"):
    if repo_url.strip():
        if not st.session_state["OPENAI_API_KEY"]:
            st.sidebar.error("❌ Please enter a valid API key before ingesting.")
        else:
            with st.spinner("Cloning & Processing Repository..."):
                repo_ingestion(repo_url)
                success = store_vectors()
            if success:
                st.sidebar.success("✅ Repo successfully ingested!")
            else:
                st.sidebar.error("❌ No valid files found in this repo.")
    else:
        st.sidebar.error("❌ Please enter a valid repo URL.")


# -------------------------------
# 📚 VECTOR STORE LOADING
# -------------------------------
persist_directory = "db"
vectordb = None
embeddings = None

if not st.session_state["OPENAI_API_KEY"]:
    st.error("❌ No API key provided. Please enter one in the sidebar before loading embeddings.")
else:
    try:
        embeddings = load_embedding()
        if os.path.exists(persist_directory):
            vectordb = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )
    except Exception as e:
        st.error("❌ Failed to initialize embeddings. Please check your API key and try again.")
        vectordb = None

# -------------------------------
# 🤖 RAG QA SYSTEM
# -------------------------------
if vectordb and st.session_state["OPENAI_API_KEY"]:
    llm = ChatOpenAI(temperature=0)
    memory = ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True
    )
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 8}
        ),
        memory=memory
    )
else:
    qa = None

# -------------------------------
# 💬 CHAT UI
# -------------------------------
st.title("🤖 Source Code Analysis using RAG")

if qa:
    if "history" not in st.session_state:
        st.session_state["history"] = []

    user_input = st.text_input("Ask a question about the repo:")

    if st.button("Submit"):
        if user_input.strip():
            with st.spinner("Generating response..."):
                response = qa(user_input)
                answer = response["answer"]
                st.session_state["history"].append((user_input, answer))

    for q, a in st.session_state["history"]:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}")
        st.markdown("---")
else:
    st.info("ℹ️ Ingest a repo and enter an API key to start chatting.")
