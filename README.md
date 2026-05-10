# 📄 Ask My PDF Bot

An AI-powered PDF chatbot built using Retrieval-Augmented Generation (RAG), semantic search, vector embeddings, and local Large Language Models.

The application allows users to upload PDFs and chat with them interactively using a local AI model powered by Ollama and Mistral.

---

# 🚀 Features

- 📄 Upload and process PDF files
- 🔍 Semantic document search
- 🧠 AI-powered question answering
- 💬 Conversational chat interface
- ⚡ Streaming responses like ChatGPT
- 📚 Source citations for answers
- 🔒 Fully local AI inference using Ollama
- 🗂️ FAISS vector database integration

---

# 🛠️ Tech Stack

- Python
- Streamlit
- FAISS
- Sentence Transformers
- Ollama
- Mistral
- NumPy
- PyMuPDF

---

# 🧠 How It Works

The application follows a Retrieval-Augmented Generation (RAG) pipeline:

1. User uploads a PDF
2. PDF text is extracted
3. Text is split into semantic chunks
4. Embeddings are generated using Sentence Transformers
5. FAISS performs vector similarity search
6. Relevant context is retrieved
7. Mistral LLM generates the final answer

---

# 📦 Installation

## Clone Repository

```bash
git clone https://github.com/codebyRamzi/ask-my-pdf-bot.git
```

## Open Project

```bash
cd ask-my-pdf-bot
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Start Ollama

```bash
ollama serve
```

## Download Mistral Model

```bash
ollama pull mistral
```

## Run Streamlit App

```bash
streamlit run streamlit_app.py
```


---

# 🔮 Future Improvements

- Multi-PDF support
- OCR support for scanned PDFs
- Hybrid search
- Conversation memory
- Cloud deployment
- Authentication system
- Advanced citation tracking

---

# 📚 Learning Outcomes

This project helped me understand:

- Retrieval-Augmented Generation (RAG)
- Vector databases
- Semantic search
- Embeddings
- Local LLM inference
- AI application architecture
- Streamlit app development

---

# 👨‍💻 Author

Mohammed Ramzi
