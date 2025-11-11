from pathlib import Path

llm_name = "mistral-small-latest"
db_dir = Path(__file__).resolve().parents[4] /'data' / 'chroma'
print(db_dir)
collection_name = "chroma_db"
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
out_name = 'shit.py'