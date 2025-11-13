from pro_solver.infer.vars.infer_vars.math_prompt_var import system_math_prompt, question_math_prompt
from pro_solver.infer.vars.infer_vars.code_prompt_var import system_code_prompt, question_code_prompt
from pro_solver.infer.vars.infer_vars.equation_var import EQUATIONS_DATASET, EQUATION_QUERIES
from pro_solver.modules.validation.code_utils import code_to_str

def equation_cfg_generate(equation_name: str) -> tuple:

    equation = EQUATIONS_DATASET[equation_name]
    math_cfg = {
                'rag_prompt': question_math_prompt,
                'rag_vars': equation,
                'user_prompt': question_math_prompt,
                'user_vars': equation,
                'system_prompt': system_math_prompt,
                'section_name': "math"
                }
    code_cfg = {
                'rag_prompt': question_code_prompt,
                'rag_vars': equation,
                'user_prompt': question_code_prompt,
                'user_vars': equation,
                'system_prompt': system_code_prompt,
                'section_name': "code"
                }
    return math_cfg, code_cfg

def equation_cfg_for_correction_generate(equation_name: str, code_to_check_name) -> tuple:
    code = code_to_str(code_to_check_name)

    equation = EQUATIONS_DATASET[equation_name]
    arxiv_cfg = {
                'rag_prompt': question_math_prompt,
                'rag_vars': equation,
                'user_prompt': question_math_prompt,
                'user_vars': equation,
                'system_prompt': system_math_prompt,
                'section_name': "arxiv"
                }

    code_to_check_cfg = {
                'rag_prompt': question_code_prompt,
                'rag_vars': equation,
                'user_prompt': question_code_prompt,
                'user_vars': equation,
                'system_prompt': ("system", str(system_code_prompt[1]) + str(code)),
                'section_name': "code"
                }
    
    query = EQUATION_QUERIES[equation_name]
    
    return arxiv_cfg, code_to_check_cfg, query