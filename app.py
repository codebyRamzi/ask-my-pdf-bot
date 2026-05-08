import fitz
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama

# =========================
# STEP 1 — READ PDF
# =========================

pdf_path = "sample.pdf"

print("\nOpening PDF...\n")

doc = fitz.open(pdf_path)

text = ""

for page in doc:
    text += page.get_text()

print("PDF Loaded Successfully\n")

# =========================
# STEP 2 — CLEAN & SPLIT TEXT
# =========================

raw_chunks = text.split("\n")

chunks = []

for chunk in raw_chunks:

    cleaned_chunk = chunk.strip()

    if len(cleaned_chunk) > 30:
        chunks.append(cleaned_chunk)

print(f"Total useful chunks created: {len(chunks)}")

if len(chunks) == 0:
    print("No valid text found in PDF!")
    exit()

# =========================
# STEP 3 — LOAD EMBEDDING MODEL
# =========================

print("\nLoading embedding model...\n")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded\n")

# =========================
# STEP 4 — CREATE EMBEDDINGS
# =========================

print("Creating embeddings...\n")

embeddings = embedding_model.encode(chunks)

embeddings = np.array(
    embeddings
).astype("float32")

print("Embeddings created successfully\n")

# =========================
# STEP 5 — CREATE FAISS INDEX
# =========================

dimension = embeddings.shape[-1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("FAISS index ready\n")

# =========================
# STEP 6 — CHAT LOOP
# =========================

while True:

    question = input(
        "\nAsk a question about the PDF (or type exit): "
    )

    if question.lower() == "exit":
        print("\nGoodbye!\n")
        break

    # =========================
    # QUESTION EMBEDDING
    # =========================

    question_embedding = embedding_model.encode([question])

    question_embedding = np.array(
        question_embedding
    ).astype("float32")

    # =========================
    # SEARCH PDF
    # =========================

    k = 3

    distances, indices = index.search(
        question_embedding,
        k
    )

    # =========================
    # RETRIEVE CONTEXT
    # =========================

    context = ""

    for idx in indices[0]:
        context += chunks[idx] + "\n"

    # =========================
    # CREATE PROMPT
    # =========================

    prompt = f"""
    You are a helpful PDF assistant.

    Answer the user's question ONLY using the PDF context below.

    PDF Context:
    {context}

    User Question:
    {question}

    Answer:
    """

    # =========================
    # OLLAMA RESPONSE
    # =========================

    response = ollama.chat(
        model="tinyllama",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    print("\n==============================")
    print("AI ANSWER")
    print("==============================\n")

    print(response["message"]["content"])

    print("\n==============================\n")