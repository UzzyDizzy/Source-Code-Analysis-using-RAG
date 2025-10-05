#store_index.py
import os
import shutil
from dotenv import load_dotenv
from langchain_chroma import Chroma
from src.helper import load_repo, text_splitter, load_embedding

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def store_vectors():
    repo_path = "repo/"
    persist_dir = "./db"

    # Recreate DB from scratch if it exists (avoid corrupted index)
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

    vectordb = Chroma.from_documents(
        text_chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    vectordb.persist()
    print("✅ Vector store created and persisted successfully!")
    return True
