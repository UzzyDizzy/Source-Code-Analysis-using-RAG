#src/helper.py
import os
import shutil
from git import Repo
import stat
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

def _remove_readonly(func, path, excinfo):
    """Fix Windows permission errors by forcing write permission."""
    os.chmod(path, stat.S_IWRITE)
    func(path)

# Clone repo
def repo_ingestion(repo_url: str):
    repo_path = "repo/"

    # --- Always nuke old repo completely ---
    if os.path.exists(repo_path):
        shutil.rmtree(repo_path, onerror=_remove_readonly)

    os.makedirs(repo_path, exist_ok=True)

    # --- Clone new repo fresh ---
    Repo.clone_from(repo_url, to_path=repo_path)
    print(f"[INFO] Repo successfully cloned into {repo_path}")


# Load all readable files (skip binaries)
def load_repo(repo_path: str):
    documents = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            file_path = os.path.join(root, file)

            # skip .git and other binary types
            if any(file.lower().endswith(ext) for ext in [
                ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico",
                ".exe", ".dll", ".so", ".dylib", ".zip", ".tar", ".gz",
                ".pdf", ".mp3", ".mp4", ".avi"
            ]):
                continue

            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                    if text.strip():
                        documents.append(Document(page_content=text, metadata={"source": file_path}))
            except Exception as e:
                print(f"⚠️ Skipping {file_path}: {e}")

    return documents


# Split into chunks
def text_splitter(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return splitter.split_documents(documents)


# Embeddings
def load_embedding():
    return OpenAIEmbeddings(disallowed_special=())
