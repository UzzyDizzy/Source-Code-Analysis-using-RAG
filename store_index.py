import os
import shutil
import streamlit as st
from langchain_chroma import Chroma
from src.helper import load_repo, text_splitter, load_embedding
from tqdm import tqdm

def store_vectors():
    OPENAI_API_KEY = st.session_state.get("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        st.error("❌ No API key available. Please provide one before storing vectors.")
        return False

    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    repo_path = "repo/"
    persist_dir = "./db"

    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir, ignore_errors=True)

    documents = load_repo(repo_path)
    if not documents:
        print("⚠️ No valid files found in repo!")
        return False

    text_chunks = text_splitter(documents)
    text_chunks = [chunk for chunk in text_chunks if len(chunk.page_content) < 4000]
    if not text_chunks:
        print("⚠️ No text chunks created!")
        return False

    embeddings = load_embedding()

    # Create empty Chroma DB (persist_directory enables automatic persistence)
    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    # Add chunks in batches
    batch_size = 50
    for i in tqdm(range(0, len(text_chunks), batch_size), desc="Embedding batches"):
        batch = text_chunks[i:i+batch_size]
        vectordb.add_documents(batch)

    # ❌ vectordb.persist() removed — not needed with latest Chroma
    # Optional manual persist if you want (internal API):
    # vectordb._client.persist()

    print("✅ Vector store created and persisted successfully!")
    return True
