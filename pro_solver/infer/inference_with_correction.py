import fire
from pro_solver.modules.rag_pipeline.base_model import LLMModel
from pro_solver.modules.rag_pipeline.full_pipeline import RagPipeline
from pro_solver.modules.collection.collection import load_collection

from pro_solver.infer.cfg_utils import equation_cfg_generate, equation_cfg_for_correction_generate
from pro_solver.infer.vars.infer_vars.model_var import llm_name, db_dir, collection_name, embedding_model, llm_correct_name
from pro_solver.database.arxiv_retriever import arxiv_to_chroma

def main(api_key: str,
         name: str,
         output_name: str,
         corrected_output_name: str
         ):
    model = LLMModel(api_key = api_key, model_name = llm_name)
    collection = load_collection(db_dir, collection_name, embedding_model)
    math_cfg, code_cfg = equation_cfg_generate(name)
    
    #----- RAG ------
    pipeline = RagPipeline(model, math_cfg, code_cfg, collection)

    pipeline(output_name)

    # ----- Correcting LLM with RAG -----
    correcting_model = LLMModel(api_key = api_key, model_name = llm_correct_name)
    arxiv_cfg, code_to_check_cfg, query = equation_cfg_for_correction_generate(name, output_name)
    arxiv_collection = arxiv_to_chroma(api_key=api_key, query=query)
    
    correction_pipeline = RagPipeline(correcting_model, arxiv_cfg, code_to_check_cfg, arxiv_collection)
    correction_pipeline(corrected_output_name)


if __name__ == "__main__":
    fire.Fire(main)
