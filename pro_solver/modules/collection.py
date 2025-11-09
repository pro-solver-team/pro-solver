import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from config.database.config import DB_DIR, COLLECTION_NAME, EMBEDDING_MODEL

def initialize_collection():
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
    return collection.count()

def query_collection(collection, query_text, n_results=3):
    return collection.query(
        query_texts=[query_text],
        n_results=n_results
    )

def load_collection(db_path: str = DB_DIR, collection_name: str = COLLECTION_NAME):
    try:
        abs_db_path = Path(db_path).resolve()
        
        if not abs_db_path.exists():
            raise FileNotFoundError(f"ChromaDB directory not found: {abs_db_path}")
        
        print(f"Loading ChromaDB from: {abs_db_path}")
        
        client = chromadb.PersistentClient(path=str(abs_db_path))
        
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embed_fn
        )
        
        print(f"✓ Collection '{collection_name}' loaded successfully")
        print(f"✓ Total documents: {collection.count()}")
        
        return collection
        
    except Exception as e:
        print(f"Error loading collection: {e}")
        raise

def print_results(results, query_text: str):
    if not results or not results["documents"]:
        print("No results found.")
        return
    
    print(f"\n🔍 Query: '{query_text}'")
    print("=" * 80)
    
    for i in range(len(results["documents"][0])):
        print(f"\n📚 Result {i+1}:")
        print(f"   ID: {results['ids'][0][i]}")
        print(f"   Distance: {results['distances'][0][i]:.4f}")
        
        if results["metadatas"] and results["metadatas"][0]:
            meta = results["metadatas"][0][i]
            print(f"   Metadata: {meta}")
        
        doc = results["documents"][0][i]
        snippet = doc[:300] + "..." if len(doc) > 300 else doc
        print(f"   Content: {snippet}")
        print("-" * 80)

def search_in_collection(query_text: str, n_results: int = 5, db_path: str = "data/chroma"):
    try:
        collection = load_collection(db_path)
        
        results = query_collection(collection, query_text, n_results)
        
        if results:
            print_results(results, query_text)
        else:
            print("No results found.")
            
        return results
        
    except Exception as e:
        print(f"Search failed: {e}")
        return None