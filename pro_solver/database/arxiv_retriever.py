from langchain_community.retrievers import ArxivRetriever
from langchain_text_splitters import CharacterTextSplitter
from pro_solver.infer.vars.infer_vars.model_var import db_dir, collection_name, embedding_model, max_docs_num
from pro_solver.modules.collection.collection import initialize_collection
import datetime

def arxiv_to_chroma(api_key: str, query: str):

    arxiv_retriever = ArxivRetriever(load_max_docs=max_docs_num)
    
    documents = arxiv_retriever.invoke(input=query)
    print(f"Retrieved {len(documents)} documents")
    
    if not documents:
        print("No documents found for the query. Nothing to add to Chroma DB.")
        return
    
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_documents = text_splitter.split_documents(documents)
    print(f"Created {len(split_documents)} chunks")
    
    if not split_documents:
        print("No text chunks were created. Nothing to add to Chroma DB.")
        return
    
    client, collection = initialize_collection(db_dir, collection_name, embedding_model)
    
    ids = [f"arxiv_{i}" for i in range(len(split_documents))]
    documents_text = [doc.page_content for doc in split_documents]
    
    metadatas = []
    for doc in split_documents:
        clean_metadata = {"section": "arxiv", "source": "arxiv"}
        
        for key, value in doc.metadata.items():
            if value is None:
                clean_metadata[key] = None
            elif isinstance(value, (str, int, float, bool)):
                clean_metadata[key] = value
            elif isinstance(value, datetime.datetime):
                clean_metadata[key] = value.isoformat()
            elif isinstance(value, datetime.date):
                clean_metadata[key] = value.isoformat()
            else:
                clean_metadata[key] = str(value)
        
        metadatas.append(clean_metadata)
    
    collection.add(
        ids=ids,
        documents=documents_text,
        metadatas=metadatas
    )
    
    print("Data for correcting LLM is successfully stored in Chroma DB")

    return collection