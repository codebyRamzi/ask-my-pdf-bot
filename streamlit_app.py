import streamlit as st
import fitz
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama
import time

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Ask My PDF Bot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Ask My PDF Bot")

st.write(
    "Chat with your PDF locally using AI."
)

# =========================
# SESSION STATE
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

# =========================
# LOAD EMBEDDING MODEL
# =========================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

embedding_model = load_embedding_model()

# =========================
# FILE UPLOAD
# =========================

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type="pdf"
)

# =========================
# PROCESS PDF
# =========================

if uploaded_file is not None:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # =========================
    # READ PDF
    # =========================

    doc = fitz.open("temp.pdf")

    text = ""

    for page in doc:
        text += page.get_text()

    # =========================
    # SMART CHUNKING
    # =========================

    chunk_size = 500
    overlap = 100

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if len(chunk.strip()) > 50:
            chunks.append(chunk)

        start += chunk_size - overlap

    # =========================
    # CREATE EMBEDDINGS
    # =========================

    embeddings = embedding_model.encode(chunks)

    embeddings = np.array(
        embeddings
    ).astype("float32")

    # =========================
    # CREATE FAISS INDEX
    # =========================

    dimension = embeddings.shape[-1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    # save in session
    st.session_state.index = index
    st.session_state.chunks = chunks

    st.success(
        f"PDF processed successfully! "
        f"{len(chunks)} chunks created."
    )

# =========================
# DISPLAY CHAT HISTORY
# =========================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# =========================
# USER INPUT
# =========================

question = st.chat_input(
    "Ask a question about your PDF..."
)

# =========================
# CHAT LOGIC
# =========================

if question:

    # save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # display user message
    with st.chat_message("user"):

        st.markdown(question)

    # check pdf uploaded
    if st.session_state.index is None:

        response_text = (
            "Please upload a PDF first."
        )

        source_text = ""

    else:

        # =========================
        # CREATE QUESTION EMBEDDING
        # =========================

        question_embedding = (
            embedding_model.encode([question])
        )

        question_embedding = np.array(
            question_embedding
        ).astype("float32")

        # =========================
        # SEARCH FAISS
        # =========================

        k = 5

        distances, indices = (
            st.session_state.index.search(
                question_embedding,
                k
            )
        )

        # =========================
        # RETRIEVE CONTEXT
        # =========================

        context = ""

        source_text = "### Sources Used:\n"

        for idx in indices[0]:

            retrieved_chunk = (
                st.session_state.chunks[idx]
            )

            context += retrieved_chunk + "\n\n"

            source_text += (
                f"- Chunk {idx}\n"
            )

        # =========================
        # BETTER PROMPT
        # =========================

        prompt = f"""
You are an intelligent PDF research assistant.

Use ONLY the provided PDF context.

Guidelines:
- Give accurate answers.
- If asked to summarize:
  provide a clean structured summary.
- Mention key points clearly.
- Avoid hallucinations.
- If answer is not in context,
  say "I could not find that in the PDF."

PDF Context:
{context}

User Question:
{question}

Helpful Answer:
"""

        # =========================
        # GENERATE RESPONSE
        # =========================

        response = ollama.chat(
            model="mistral",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        response_text = (
            response["message"]["content"]
        )

    # =========================
    # STREAMING EFFECT
    # =========================

    with st.chat_message("assistant"):

        message_placeholder = st.empty()

        full_response = ""

        for word in response_text.split():

            full_response += word + " "

            time.sleep(0.02)

            message_placeholder.markdown(
                full_response + "▌"
            )

        message_placeholder.markdown(
            full_response
        )

        st.markdown(source_text)

    # =========================
    # SAVE ASSISTANT MESSAGE
    # =========================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content":
            full_response
            + "\n\n"
            + source_text
        }
    )