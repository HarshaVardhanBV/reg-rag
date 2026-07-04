import fitz  # PyMuPDF
import chromadb
from chromadb.utils import embedding_functions


def load_pdf(path: str) -> str:
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split by words with overlap so context doesn't get cut at boundaries."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def ingest(pdf_path: str, collection_name: str = "regulatory_docs"):
    client = chromadb.PersistentClient(path="./chroma_db")

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Delete old collection if re-running
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        embedding_function=ef,
    )

    print(f"Loading {pdf_path}...")
    text = load_pdf(pdf_path)
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks")

    collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))],
    )
    print(f"✅ Ingested {len(chunks)} chunks into ChromaDB")


if __name__ == "__main__":
    ingest("data/ich_e2a.pdf")