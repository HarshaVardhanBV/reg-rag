import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="./chroma_db")

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.get_collection(
    name="regulatory_docs",
    embedding_function=ef,
)

question = "What are the criteria for a serious adverse event?"

results = collection.query(
    query_texts=[question],
    n_results=3,
)

print(f"Question: {question}\n")
for i, doc in enumerate(results["documents"][0]):
    print(f"--- Chunk {i+1} ---")
    print(doc[:400])
    print()