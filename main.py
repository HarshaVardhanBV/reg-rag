from dotenv import load_dotenv
load_dotenv(override=True)

import os
import chromadb
from chromadb.utils import embedding_functions
import anthropic
import fitz
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- Embedding setup ---
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

def load_pdf(path: str) -> str:
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + chunk_size]))
        i += chunk_size - overlap
    return chunks

def run_ingest():
    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        client.delete_collection("regulatory_docs")
    except Exception:
        pass
    collection = client.create_collection(name="regulatory_docs", embedding_function=ef)
    text = load_pdf("data/ich_e2a.pdf")
    chunks = chunk_text(text)
    collection.add(documents=chunks, ids=[f"chunk_{i}" for i in range(len(chunks))])
    print(f"✅ Ingested {len(chunks)} chunks")

# --- FastAPI lifespan: runs ingest once at startup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global collection
    print("Starting up — ingesting documents...")
    run_ingest()
    collection = chroma_client.get_collection(
        name="regulatory_docs", embedding_function=ef
    )
    yield
    # nothing to clean up

app = FastAPI(title="Regulatory RAG API", lifespan=lifespan)

# --- Clients ---
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = None
#chroma_client.get_or_create_collection(
 #   name="regulatory_docs", embedding_function=ef
#)
claude = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# --- Pydantic models ---
class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    sources: list[str]

# --- System prompt ---
SYSTEM_PROMPT = """You are a regulatory affairs specialist with expertise in ICH and FDA guidelines.
Answer the question using ONLY the context provided below.
If the answer is not in the context, say exactly: "I don't have enough information in the provided documents."
Always cite which source number supports your answer (e.g. "According to Source 1...")."""

# --- RAG function ---
def rag_ask(question: str, n_results: int = 3) -> dict:
    results = collection.query(query_texts=[question], n_results=n_results)
    chunks = results["documents"][0]
    context = "\n\n---\n\n".join(
        f"[Source {i+1}]\n{chunk}" for i, chunk in enumerate(chunks)
    )
    response = claude.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}]
    )
    return {"answer": response.content[0].text, "sources": chunks}

# --- Endpoints ---
@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    return AskResponse(**rag_ask(request.question))

@app.get("/")
def root():
    return {"status": "Regulatory RAG API running", "docs": "/docs"}