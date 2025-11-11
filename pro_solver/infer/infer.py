import fire
from pro_solver.modules.rag_pipeline.base_model import LLMModel
from pro_solver.modules.rag_pipeline.full_pipeline import RagPipeline
from pro_solver.modules.collection.collection import load_collection
from pro_solver.infer.vars.infer_vars.model_var import LLM_NAME, TEMPERATURE, DB_DIR, EMBEDDING_MODEL, COLLECTION_NAME, OUTPUT_SOLVER_NAME
from pro_solver.infer.vars.util_vars.math_vars import math_config
from pro_solver.infer.vars.util_vars.code_vars import code_config

# INVALID CODE CHECK OUT MAIN_AGENT_CORRECTION BRANCH
def main(api_key: str) -> None:
    model = LLMModel(api_key=api_key, model_name=LLM_NAME, temperature=TEMPERATURE)
    collection = load_collection(db_path=DB_DIR, collection_name=COLLECTION_NAME, embedding_model=EMBEDDING_MODEL)

    #----- RAG ------
    pipeline = RagPipeline(model, math_config, code_config, collection)

    pipeline(OUTPUT_SOLVER_NAME)

if __name__ == "__main__":
    fire.Fire(main)
