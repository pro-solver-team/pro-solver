import chromadb
from chromadb.utils import embedding_functions
from pro_solver.modules.config import DB_DIR, COLLECTION_NAME, EMBEDDING_MODEL

def initialize_chroma_client():
    """Initialize ChromaDB client and collection"""
    client = chromadb.PersistentClient(path=DB_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}
    )
    
    print(f"Chroma path: {DB_DIR}")
    print(f"Collection: {COLLECTION_NAME}")
    return client, collection

def get_collection_count(collection):
    """Get the number of documents in the collection"""
    return collection.count()

def query_collection(collection, query_text, n_results=3):
    """Query the collection and return results"""
    return collection.query(
        query_texts=[query_text],
        n_results=n_results
    )