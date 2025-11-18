from pro_solver.infer.vars.infer_vars.math_prompt_var import system_math_prompt, question_math_prompt
from pro_solver.infer.vars.infer_vars.code_prompt_var import system_code_prompt, question_code_prompt
from pro_solver.infer.vars.infer_vars.code_analyzer_prompt_var import system_code_anal_prompt, question_code_anal_prompt
from pro_solver.infer.vars.infer_vars.equation_var import EQUATIONS_DATASET

def equation_cfg_generate() -> tuple:
    math_model_cfg = {
                     'rag_prompt': question_math_prompt,
                     'user_prompt': question_math_prompt,
                     'system_prompt': system_math_prompt,
                     'section_name': "math"
                     }

    code_model_cfg = {
                    'rag_prompt': question_code_prompt,
                    'user_prompt': question_code_prompt,
                    'system_prompt': system_code_prompt,
                    'section_name': "code"
                    }

    code_anal_model_cfg = {
                            'rag_prompt': question_code_anal_prompt,
                            'system_prompt': system_code_anal_prompt,
                            'user_prompt': question_code_anal_prompt,
                            'section_name': "code"
                            }
    return math_model_cfg, code_model_cfg, code_anal_model_cfg

def get_equation(equation_name: str):
    return EQUATIONS_DATASET[equation_name]