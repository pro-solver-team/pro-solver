from pathlib import Path

llm_name = "mistral-small-2501"
# db_dir = Path(__file__).resolve().parents[4] /'data' / 'chroma'
db_dir = "../../data/chroma"
collection_name = "chroma_db"
out_name = 'shit'
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

llm_correct_name = "mistral-small-2501"
max_docs_num = 10
