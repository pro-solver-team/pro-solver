import hydra
from omegaconf import DictConfig

from pro_solver.modules.collection.collection import initialize_collection, get_collection_count
from pro_solver.modules.collection.dataset_load.dataset_load import upsert_dataset
from pro_solver.modules.collection.repo_load.github_process import add_repos_to_chroma
from pro_solver.modules.collection.repo_load.vars import DATASETS, ALL_REPOS, FINITE_DIFF_REPOS

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(config: DictConfig) -> None:
    client, collection = initialize_collection(config["database"]["db_dir"],
                                               config["database"]["collection_name"],
                                               config["database"]["embedding_model"])
    
    for repo in DATASETS:
        upsert_dataset(collection, repo, config["database"]["max_records_per_dataset"])
    
    add_repos_to_chroma(collection, FINITE_DIFF_REPOS) # CHANGE TO ALL_REPOS FOR INFERENCE
    
    print(f"Total documents in collection: {get_collection_count(collection)}")


if __name__ == "__main__":
    main()
