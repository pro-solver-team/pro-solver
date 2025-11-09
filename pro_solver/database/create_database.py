from pro_solver.modules.collection import initialize_collection, query_collection, get_collection_count
from pro_solver.modules.dataset_load import upsert_dataset
from pro_solver.modules.github_process import add_repos_to_chroma
from config.database.config import DATASETS, MAX_RECORDS_PER_DATASET, FINITE_DIFF_GITHUB_REPOS

def main():
    client, collection = initialize_collection()
    
    for repo in DATASETS:
        upsert_dataset(collection, repo, MAX_RECORDS_PER_DATASET)
    
    add_repos_to_chroma(collection, FINITE_DIFF_GITHUB_REPOS)
    
    print(f"Total documents in collection: {get_collection_count(collection)}")


if __name__ == "__main__":
    main()
