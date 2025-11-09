from pro_solver.modules.collection import initialize_chroma_client, query_collection, get_collection_count
from pro_solver.modules.dataset_load import upsert_dataset
from pro_solver.modules.github_process import add_repos_to_chroma
from pro_solver.modules.config import DATASETS, MAX_RECORDS_PER_DATASET, FINITE_DIFF_GITHUB_REPOS

def main():
    client, collection = initialize_chroma_client()
    
    for repo in DATASETS:
        upsert_dataset(collection, repo, MAX_RECORDS_PER_DATASET)
    
    add_repos_to_chroma(collection, FINITE_DIFF_GITHUB_REPOS)
    
    results = query_collection(collection, "Dirichlet boundary conditions", n_results=5)
    
    print(f"Total documents in collection: {get_collection_count(collection)}")
    
    return collection

if __name__ == "__main__":
    main()
