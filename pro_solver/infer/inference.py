import fire
from pro_solver.modules.rag_pipeline.base_model import LLMModel
from pro_solver.modules.rag_pipeline.full_pipeline import RagPipeline
from pro_solver.modules.collection.collection import load_collection

from pro_solver.infer.cfg_utils import equation_cfg_generate, get_equation
from pro_solver.infer.vars.infer_vars.model_var import llm_code_name, llm_math_name, db_dir, collection_name, embedding_model

def main(api_key: str,
         name: str,
         rag: bool,
         output_name: str
         ):
    model_code = LLMModel(api_key = api_key,  model_name = llm_code_name, temperature=0.3)
    model_math = LLMModel(api_key=api_key, model_name=llm_math_name, temperature=0.2)

    math_cfg, code_cfg, code_anal_cfg = equation_cfg_generate()
    math_cfg['rag'] = rag
    code_cfg['rag'] = False
    code_anal_cfg['rag'] = rag

    equation = get_equation(name)
    ###VARS
    math_vars = {
                 'math_user_vars': equation,
                 'math_rag_vars': equation
                 }
    code_vars = {
        'code_user_vars': equation,
        'code_rag_vars': equation
    }

    if rag:
        collection = load_collection(db_dir, collection_name, embedding_model)
        pipeline = RagPipeline(model_code, model_math, math_cfg, code_cfg, code_anal_cfg, collection)
    else:
        pipeline = RagPipeline(model_code, math_cfg, code_cfg, code_anal_cfg, None)
    #----- RAG ------
    pipeline(output_name, **math_vars, **code_vars)

if __name__ == "__main__":
    fire.Fire(main)
