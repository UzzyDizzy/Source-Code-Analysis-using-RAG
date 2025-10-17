import os
import shutil
import streamlit as st
from langchain_chroma import Chroma
from src.helper import load_repo, text_splitter, load_embedding

def store_vectors():
    # ✅ Use runtime key from session
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
    if not text_chunks:
        print("⚠️ No text chunks created!")
        return False

    embeddings = load_embedding()

    # Safety: Filter out chunks that are too large
    text_chunks = [chunk for chunk in text_chunks if len(chunk.page_content) < 6000]


    vectordb = Chroma.from_documents(
        text_chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    vectordb.persist()
    print("✅ Vector store created and persisted successfully!")
    return True
