#app.py
import os
import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from src.helper import repo_ingestion, load_embedding
from store_index import store_vectors

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# Sidebar - repo ingestion
st.sidebar.title("⚡ Configurations")
repo_url = st.sidebar.text_input("Enter GitHub Repo URL:")

if st.sidebar.button("Ingest Repo"):
    if repo_url.strip():
        with st.spinner("Cloning & Processing Repository..."):
            repo_ingestion(repo_url)
            success = store_vectors()
        if success:
            st.sidebar.success("✅ Repo successfully ingested!")
        else:
            st.sidebar.error("❌ No valid files found in this repo.")
    else:
        st.sidebar.error("❌ Please enter a valid repo URL.")


# Load vectorDB
embeddings = load_embedding()
persist_directory = "db"

if os.path.exists(persist_directory):
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
else:
    vectordb = None


# Setup QA chain
if vectordb:
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


# Chat UI
st.title("🤖 Source Code Analysis using RAG")

if qa:
    # Display conversation history
    if "history" not in st.session_state:
        st.session_state["history"] = []

    user_input = st.text_input("Ask a question about the repo:")

    if st.button("Submit"):
        if user_input.strip():
            with st.spinner("Generating response..."):
                response = qa(user_input)
                answer = response["answer"]

                # Save history
                st.session_state["history"].append((user_input, answer))

    # Show chat history
    for q, a in st.session_state["history"]:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}")
        st.markdown("---")

else:
    st.info("ℹ️ Ingest a repo first from the sidebar.")
