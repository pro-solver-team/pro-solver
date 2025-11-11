import hydra
from omegaconf import DictConfig
import fire
from pro_solver.modules.rag_pipeline.base_model import LLMModel
from pro_solver.modules.rag_pipeline.full_pipeline import RagPipeline
from pro_solver.modules.collection.collection import load_collection


@hydra.main(version_base=None, config_path="../../config", config_name="infer/config")
def main(cfg: DictConfig):
    model = LLMModel(api_key = cfg.api_key, model_name = cfg.llm_name)
    collection = load_collection(cfg.db_dir, cfg.collection_name)

    #----- RAG ------
    rag_dict = cfg.math_config
    code_dict = cfg.code_config
    pipeline = RagPipeline(model, rag_dict, code_dict, collection)

    pipeline(cfg.output_solver_name)

if __name__ == "__main__":
    fire.Fire()
