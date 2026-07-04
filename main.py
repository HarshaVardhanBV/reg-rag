from dotenv import load_dotenv
load_dotenv(override=True)

import os
import chromadb
from chromadb.utils import embedding_functions
import anthropic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Regulatory RAG API")

# --- Clients (created once at startup) ---
chroma_client = chromadb.PersistentClient(path="./chroma_db")
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
collection = chroma_client.get_collection(name="regulatory_docs", embedding_function=ef)
claude = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# --- Pydantic models ---
class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    sources: list[str]

# --- RAG system prompt ---
SYSTEM_PROMPT = """You are a regulatory affairs specialist with expertise in ICH and FDA guidelines.
Answer the question using ONLY the context provided below.
If the answer is not in the context, say exactly: "I don't have enough information in the provided documents."
Always cite which source number supports your answer (e.g. "According to Source 1...")."""

# --- Core RAG function ---
def rag_ask(question: str, n_results: int = 3) -> dict:
    # Step 1: Retrieve relevant chunks
    results = collection.query(query_texts=[question], n_results=n_results)
    chunks = results["documents"][0]

    # Step 2: Format chunks as numbered sources
    context = "\n\n---\n\n".join(
        f"[Source {i+1}]\n{chunk}" for i, chunk in enumerate(chunks)
    )

    # Step 3: Call Claude
    response = claude.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }]
    )

    return {
        "answer": response.content[0].text,
        "sources": chunks
    }

# --- Endpoints ---
@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    result = rag_ask(request.question)
    return AskResponse(**result)

@app.get("/")
def root():
    return {"status": "Regulatory RAG API running", "docs": "/docs"}