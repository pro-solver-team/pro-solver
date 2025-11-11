from pathlib import Path

llm_name = "mistral-small-latest"
db_dir = Path(__file__).resolve().parents[4] /'data' / 'chroma'
collection_name = "chroma_db"
out_name = 'shit'
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"