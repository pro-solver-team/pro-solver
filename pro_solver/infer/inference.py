import hydra
from omegaconf import DictConfig
from pathlib import Path

from pro_solver.modules.rag_pipeline.base_model import LLMModel
from pro_solver.modules.rag_pipeline.full_pipeline import RagPipeline
from pro_solver.modules.collection.collection import load_collection

root_path = Path(__file__).resolve().parents[2]
config_rel_path = "config/infer"
config_path = root_path/config_rel_path
config_name = "config.yaml"
@hydra.main(config_path=str(config_path), config_name=str(config_name))
def main(cfg: DictConfig):
    model = LLMModel(api_key = cfg.MISTRAL_API_KEY, model_name = cfg.LLM_NAME)
    collection = load_collection(DB_DIR, COLLECTION_NAME)

    #----- RAG ------

    rag_dict = cfg.math
    print(rag_dict)
    code_dict = cfg.code
    pipeline = RagPipeline(model, rag_dict, code_dict, collection)

    pipeline('shit.py')

if __name__ == "__main__":
    main()
    #fire.Fire(main)