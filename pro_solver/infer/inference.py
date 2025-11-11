import fire
from pro_solver.modules.rag_pipeline.base_model import LLMModel
from pro_solver.modules.rag_pipeline.full_pipeline import RagPipeline
from pro_solver.modules.collection.collection import load_collection

from pro_solver.infer.cfg_utils import equation_cfg_generate
from pro_solver.infer.vars.infer_vars.model_var import llm_name, db_dir, collection_name, embedding_model

def main(api_key: str,
         name: str,
         output_name: str
         ):
    model = LLMModel(api_key = api_key, model_name = llm_name)
    collection = load_collection(db_dir, collection_name, embedding_model)
    math_cfg, code_cfg = equation_cfg_generate(name)
    #----- RAG ------
    pipeline = RagPipeline(model, math_cfg, code_cfg, collection)

    pipeline(output_name)

if __name__ == "__main__":
    fire.Fire(main)
