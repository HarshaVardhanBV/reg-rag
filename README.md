# Regulatory RAG Q&A

A Retrieval-Augmented Generation (RAG) system for querying ICH/FDA regulatory guidelines. Ask questions in plain English — get cited answers grounded in the actual guideline text. No hallucination, no guessing.

**Live API:** https://reg-rag-production.up.railway.app

---

## The problem

Regulatory teams waste hours manually searching dense ICH/FDA guidelines for specific criteria. Traditional keyword search misses semantically related content. A generic LLM might hallucinate regulatory criteria — unacceptable in a compliance context.

This API retrieves the exact guideline text that answers your question, passes it to Claude as numbered sources, and returns a cited answer. If the answer isn't in the documents, it says so.

---

## How it works

```
Question
  → Embed (sentence-transformers / all-MiniLM-L6-v2)
  → ChromaDB similarity search → top-3 chunks
  → Claude: "Answer ONLY from the provided context"
  → Answer with [Source N] citations + raw source chunks
```

1. **Ingest** (runs once at startup): ICH E2A PDF → 500-word chunks with 50-word overlap → embedded → stored in ChromaDB on disk
2. **Retrieve**: question embedded → 3 nearest chunks returned by cosine similarity
3. **Generate**: Claude receives chunks as `[Source 1]`, `[Source 2]`, `[Source 3]` with a strict grounding instruction → returns cited answer

---

## Stack

| Layer | Tool |
|---|---|
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) — free, local |
| Vector store | ChromaDB (`PersistentClient`) — disk-backed, no server |
| PDF parsing | PyMuPDF (`fitz`) |
| LLM | Claude (`claude-opus-4-8`) via Anthropic SDK |
| API | FastAPI + Pydantic |
| UI | Streamlit |
| Deploy | Railway |

---

## Endpoints

### `POST /ask`

```bash
curl -X POST https://reg-rag-production.up.railway.app/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the criteria for a serious adverse event?"}'
```

Response:
```json
{
  "answer": "According to Source 2, a serious adverse event (experience) or reaction is any untoward medical occurrence that at any dose: results in death, is life-threatening, requires inpatient hospitalisation or prolongation of existing hospitalisation, results in persistent or significant disability/incapacity, or is a congenital anomaly/birth defect...",
  "sources": [
    "chunk 1 text...",
    "chunk 2 text...",
    "chunk 3 text..."
  ]
}
```

### `GET /`

```bash
curl https://reg-rag-production.up.railway.app/
```

### Interactive docs

https://reg-rag-production.up.railway.app/docs

---

## Local setup

```bash
git clone https://github.com/HarshaVardhanBV/reg-rag.git
cd reg-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` (never commit this):
```
ANTHROPIC_API_KEY=sk-ant-...
```

The app ingests documents automatically on startup. Just run:

```bash
# Terminal 1 — API
uvicorn main:app --reload

# Terminal 2 — UI
streamlit run app.py
```

- API: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- Streamlit: http://localhost:8501

---

## Repo structure

```
reg-rag/
├── data/
│   └── ich_e2a.pdf          # ICH E2A guideline (committed to repo)
├── chroma_db/               # vector store — gitignored, rebuilt on startup
├── ingest.py                # PDF → chunks → ChromaDB (also called at startup)
├── query.py                 # dev testing of retrieval only
├── main.py                  # FastAPI app: RAG chain + /ask endpoint
├── app.py                   # Streamlit UI
├── Procfile                 # Railway: web: uvicorn main:app --host 0.0.0.0 --port $PORT
├── requirements.txt
└── .env                     # gitignored — ANTHROPIC_API_KEY here
```

---

## Design decisions

- **"ONLY from context" system prompt** — prevents Claude from answering from training data. If the answer isn't in the retrieved chunks, Claude says so explicitly. Critical for regulatory use where a hallucinated criterion could have compliance consequences.
- **Numbered sources** — chunks formatted as `[Source 1]`, `[Source 2]` so Claude can cite specific passages. Raw source chunks returned alongside the answer for human verification.
- **Ingest on startup** — Railway's filesystem resets on every redeploy. Re-ingesting 9 chunks at startup (~30s) guarantees a fresh vector store. Production alternative: managed vector DB (Pinecone, Weaviate).
- **sentence-transformers over API embeddings** — free, local, no extra API key. Adequate for a small regulatory doc corpus. Switch to `voyage-3` at scale.
- **ChromaDB over FAISS** — `PersistentClient` saves to disk automatically, simpler API. FAISS is faster at scale but requires manual save/load.
- **Streamlit over React** — RAG Q&A is a single-interaction flow. Streamlit is ~30 lines vs 300+ for React. Right tool for a demo.