from omegaconf import DictConfig

from pro_solver.modules.rag_pipeline.base_pipeline import ModelPipeline
from pro_solver.modules.rag_pipeline.base_model import LLMModel
from pro_solver.modules.validation.output_scheme import PDEOutput
from pro_solver.modules.collection.dataset_load.text_process import safe_json_parse

from pro_solver.modules.validation.code_utils import code_save, code_check

from chromadb.api.models.Collection import Collection


class RagPipeline:
    def __init__(self, model: LLMModel,
                 math_cfg: DictConfig,
                 code_cfg: DictConfig,
                 db: Collection,
                 info_num: int = 5
                 ):
        self.model = model
        self.math_pipeline = ModelPipeline(**math_cfg)
        self.code_pipeline = ModelPipeline(**code_cfg)
        self.collection = db
        self.page_num = info_num

    def __call__(self, name):
        math_context = self.math_pipeline.generate_response(self.collection, self.page_num)
        while (True):
            code_text = self.code_pipeline.generate_response(self.collection, math_context, self.page_num)
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