from omegaconf import DictConfig

from pro_solver.modules.rag_pipeline.base_pipeline import ModelPipeline
from pro_solver.modules.rag_pipeline.base_model import LLMModel
from pro_solver.modules.validation.output_scheme import PDEOutput
from pro_solver.modules.collection.dataset_load.text_process import safe_json_parse

from pro_solver.modules.validation.code_utils import code_save, code_check
from typing import Optional
from chromadb.api.models.Collection import Collection


class RagPipeline:
    def __init__(self,
                 model_code: LLMModel,
                 model_math: LLMModel,
                 math_cfg: DictConfig,
                 code_cfg: DictConfig,
                 code_anal_cfg: DictConfig,
                 db: Optional[Collection],
                 info_num_math: int = 5,
                 info_num_code: int = 5
                 ):
        self.math_model = model_math
        self.code_model = model_code
        self.math_pipeline = ModelPipeline(self.math_model, **math_cfg)
        self.code_anal_pipeline = ModelPipeline(self.math_model, **code_anal_cfg)
        self.code_pipeline = ModelPipeline(self.code_model, **code_cfg)
        self.collection = db
        self.page_code_num = info_num_code
        self.page_math_num = info_num_math

    def __call__(self,
                 name: str,
                 math_rag_vars: dict,
                 math_user_vars: dict,
                 code_rag_vars: dict,
                 code_user_vars: dict
                 ):
        math_context = self.math_pipeline.generate_response(
                                                            db=self.collection,
                                                            num_res=self.page_math_num,
                                                            rag_vars=math_rag_vars,
                                                            user_vars=math_user_vars,
                                                            add_info=None,
                                                            )

        structured_code = self.code_anal_pipeline.generate_response(
                                                                    db=self.collection,
                                                                    num_res=self.page_code_num,
                                                                    rag_vars=math_rag_vars,
                                                                    user_vars=math_user_vars,
                                                                    add_info=math_context,
                                                                    )


        code_user_vars['math_context'] = math_context
        code_user_vars['code_context'] = structured_code
       # print(math_context, structured_code)
        while (True):
            code_text = self.code_pipeline.generate_response(
                                                             #db=self.collection,
                                                             num_res=self.page_code_num,
                                                           #  rag_vars=code_rag_vars,
                                                             user_vars=code_user_vars,
                                                             #add_info=math_context
                                                             )
            try:
                code_json = safe_json_parse(code_text)
            except:
                continue
            pde_output = PDEOutput(**code_json)

            full_code = "\n\n".join([
                pde_output.function,
                pde_output.example
            ])
            final_code = "\n\n".join([
                pde_output.function])

            if not code_check(full_code):
                code_save(final_code, name)
                break