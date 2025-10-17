import os
import shutil
import streamlit as st
from langchain_chroma import Chroma
from src.helper import load_repo, text_splitter, load_embedding
from tqdm import tqdm

def store_vectors():
    """
    Load repo documents, split them into chunks, embed using OpenAI embeddings,
    and store in a Chroma vector database with persistence.
    Returns True if successful, False otherwise.
    """

    # 1️⃣ Get OpenAI API key from Streamlit session state
    OPENAI_API_KEY = st.session_state.get("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        st.error("❌ No API key available. Please provide one before storing vectors.")
        return False

    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    # 2️⃣ Define repo path & persistence directory
    repo_path = "repo/"
    persist_dir = "./db"

    # Remove previous DB to avoid readonly issues
    if os.path.exists(persist_dir):
        try:
            shutil.rmtree(persist_dir, ignore_errors=True)
        except Exception as e:
            st.warning(f"⚠️ Could not delete old DB folder: {e}")

    # 3️⃣ Load documents from repo
    documents = load_repo(repo_path)
    if not documents:
        st.warning("⚠️ No valid files found in repo!")
        return False

    # 4️⃣ Split into chunks, filter too-large ones
    text_chunks = text_splitter(documents)
    text_chunks = [chunk for chunk in text_chunks if len(chunk.page_content) < 4000]
    if not text_chunks:
        st.warning("⚠️ No text chunks created!")
        return False

    # 5️⃣ Load embeddings
    embeddings = load_embedding()

    # 6️⃣ Create Chroma DB with persistence
    try:
        vectordb = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
    except Exception as e:
        st.error(f"❌ Failed to create Chroma DB: {e}")
        return False

    # 7️⃣ Embed chunks in batches for safety
    batch_size = 50
    try:
        for i in tqdm(range(0, len(text_chunks), batch_size), desc="Embedding batches"):
            batch = text_chunks[i:i+batch_size]
            vectordb.add_documents(batch)
    except Exception as e:
        st.error(f"❌ Error adding documents to Chroma: {e}")
        return False

    # ✅ Chroma automatically persists if persist_directory is set
    st.success("✅ Vector store created and persisted successfully!")
    return True
