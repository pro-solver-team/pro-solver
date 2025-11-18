from chromadb.api.models.Collection import Collection
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from pro_solver.modules.rag_pipeline.pde_prompt import PDEPPrompt
from pro_solver.modules.rag_pipeline.base_model import LLMModel

class ModelPipeline():
  def __init__(self,
               model: LLMModel,
               rag: bool,
               system_prompt: tuple,
               section_name: str,
               user_prompt: tuple,
               rag_prompt: str = None
               ):
    self.llm = model
    self.rag = rag
    if self.rag:
        self.rag_temp = ChatPromptTemplate.from_messages(rag_prompt)
        #self.rag_vars = rag_vars
    self.system_prompt = system_prompt
    #self.user_vars = user_vars
    self.user_prompt = user_prompt
    self.section_name = section_name

  def search_rag_res(self,
                     db: Collection,
                     rag_vars: dict,
                     num_res: int,
                     additional_info: str = None,
                     cat_info: bool = False
                     ):
      message = self.rag_temp.format_messages(**rag_vars)
      if additional_info:
          message[1].content += "\n" + additional_info
      results = db.query(
                         query_texts=[message[1].content],
                         n_results=num_res,
                         where={"section": self.section_name}
                        )
      return " ".join(results['documents'][0])


  def generate_prompt(self):
    return PDEPPrompt(self.system_prompt, self.user_prompt, context = self.rag).template()


  def generate_response(self,
                        user_vars: dict,
                        num_res: int,
                        rag_vars: dict = None,
                        db: Optional[Collection] = None,
                        add_info: dict = None
                        ):
    if self.rag and not db:
        raise ValueError('no collection given')

    if self.rag:
        if not add_info:
          rag_context = self.search_rag_res(
                                            db=db,
                                            num_res=num_res,
                                            cat_info=False,
                                            rag_vars=rag_vars
                                            )
        else:
          rag_context = self.search_rag_res(
                                            db=db,
                                            num_res=num_res,
                                            additional_info=add_info,
                                            rag_vars=rag_vars
                                            )
        rag_vars['context'] = rag_context
        full_request = {
                        **user_vars,
                        **rag_vars
                        }

    else:
        full_request = {
            **user_vars
        }
    prompt_temp = self.generate_prompt()
    print(full_request)
    return self.llm(prompt_temp, full_request).content