# 🤖 Source Code Analysis using RAG

A powerful Streamlit-based application that uses **Retrieval-Augmented Generation (RAG)** with **LangChain**, **OpenAI LLMs**, and **ChromaDB** to analyze and understand source code from any GitHub repository.  
It allows you to clone a repository, index its codebase into a vector database, and interact with it conversationally — asking natural language questions and receiving intelligent answers.

---

## 🚀 Features

- 🧠 **LLM-Powered Code Understanding** – Ask natural language questions about the codebase and get context-aware answers.
- 📁 **Automatic Repo Ingestion** – Clone any GitHub repository directly from the sidebar.
- 📚 **Chunking & Embedding** – Split large code files into manageable chunks and embed them using OpenAI.
- 🔍 **Semantic Search** – Retrieve relevant parts of the repository for each question.
- 💬 **Conversational Memory** – Keeps track of previous questions and answers for a smooth Q&A experience.
- 🔐 **Dynamic API Key Input** – Enter your OpenAI API key securely from the sidebar (clears on refresh).

---

## 🛠️ Tech Stack

- [Python](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [ChromaDB](https://www.trychroma.com/)
- [OpenAI API](https://platform.openai.com/)
- [GitPython](https://gitpython.readthedocs.io/)

---

## 📂 Project Structure

```
Source-Code-Analysis-with-RAG/
│
├── app.py # 🎯 Main Streamlit app
├── store_index.py # 🧠 Stores code embeddings into ChromaDB
│
├── src/ # 📁 Helper modules
│ └── helper.py # 📦 Repo ingestion, text splitting, embeddings
│
├── repo/ # 📂 Cloned GitHub repo (auto-created)
│
├── db/ # 📊 Chroma vector database (auto-generated)
│
├── secrets.toml # 🔑 Optional secrets storage (ignored in .gitignore)
├── requirements.txt # 📦 Project dependencies
├── README.md # 📘 Project documentation
└── .gitignore # 🙈 Ignored files (db/, repo/, secrets.toml, etc.)
```

---

## ⚙️ Setup Instructions

### 1. 📥 Clone this Repository

```
git clone https://github.com/yourusername/Source-Code-Analysis-with-RAG.git
cd Source-Code-Analysis-with-RAG
```

### 2. 📦 Create a Virtual Environment (Recommended)

```
python -m venv myenv
source myenv/bin/activate   # On Windows: myenv\Scripts\activate
```

### 3. 📚 Install Dependencies
```
pip install -r requirements.txt
```

### 4. 🔐 Configure Your API Key

You have two ways to provide your OpenAI API Key:

✅ Option 1: Use secrets.toml (recommended for Streamlit Cloud)
Create a file: .streamlit/secrets.toml
```
[openai]
api_key = "sk-proj-XXXXXXXXXXXXXXXXXXXXXXXX"
```
✅ Option 2: Enter API Key via Sidebar

No need to store it locally — just paste it in the sidebar when running the app.
The key is cleared every time you refresh the page or enter a new one.

### 5. ▶️ Run the Streamlit App
```
streamlit run app.py
```

---

## 🧪 How It Works

1. Enter a GitHub repository URL in the sidebar.

2. The repo is cloned and all text-based source files are extracted.

3. The code is chunked and embedded using OpenAI embeddings.

4. The embeddings are stored in a local Chroma vector database.

5. Ask any question about the codebase — the system retrieves relevant chunks and generates a response using a RAG pipeline.

---

## 📸 Example Usage

- “What does the main function do in this repo?”

- “Which file handles authentication?”

- “Explain how the database connection is managed.”

- “Where is the API key validated?”

## 📌 Notes

- Ensure that your OpenAI API key has access to gpt-4 or gpt-3.5 models.

- The repo/ and db/ folders are automatically recreated whenever you ingest a new repository.

- Each session is isolated — memory and keys reset on refresh.

## 📜 License

This project is licensed under the MIT License.
Feel free to use and modify it for your own projects!

## 🤝 Contributing

Contributions are welcome!

- Fork the repository

- Create a new branch (feature/my-feature)

- Commit your changes

- Push and open a pull request 🚀

## 🧠 Inspiration

This project was built to demonstrate how RAG (Retrieval-Augmented Generation) can be applied beyond documents — enabling natural language interaction with source code, unlocking powerful code review, documentation, and analysis capabilities.