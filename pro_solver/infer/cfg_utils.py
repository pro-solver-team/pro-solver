from pro_solver.infer.vars.infer_vars.math_prompt_var import system_math_prompt, question_math_prompt
from pro_solver.infer.vars.infer_vars.code_prompt_var import system_code_prompt, question_code_prompt
from pro_solver.infer.vars.infer_vars.equation_var import EQUATIONS_DATASET
#import fire

def equation_cfg_generate(equation_name: str) -> tuple:

    equation = EQUATIONS_DATASET[equation_name]
    math_cfg = {
                'rag_prompt': question_math_prompt,
                'rag_vars': equation,
                'user_prompt': question_math_prompt,
                'user_vars': equation,
                'system_prompt': question_math_prompt,
                'section_name': "math"
                }
    code_cfg = {
                'rag_prompt': question_code_prompt,
                'rag_vars': equation,
                'user_prompt': question_code_prompt,
                'user_vars': equation,
                'system_prompt': question_code_prompt,
                'section_name': "code"
                }
    return math_cfg, code_cfg


#if __name__ == "__main__":
#    fire.Fire(model_cfg_generate)

