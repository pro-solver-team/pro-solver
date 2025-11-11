from chromadb.api.models.Collection import Collection
from langchain_core.prompts import ChatPromptTemplate
from pro_solver.modules.rag_pipeline.pde_prompt import PDEPPrompt
from pro_solver.modules.rag_pipeline.base_model import LLMModel

class ModelPipeline():
  def __init__(self,
               model: LLMModel,
               rag_prompt: str,
               rag_vars: dict,
               user_prompt: tuple,
               user_vars: dict,
               system_prompt: tuple,
               section_name: str
               ):
    self.llm = model
    self.rag_temp = ChatPromptTemplate.from_messages(rag_prompt)
    self.rag_vars = rag_vars
    self.system_prompt = system_prompt
    self.user_vars = user_vars
    self.user_prompt = user_prompt
    self.section_name = section_name

  def search_rag_res(self,
                     db: Collection,
                     num_res: int,
                     additional_info: str = None):

    message = self.rag_temp.format_messages(**self.rag_vars)
    if additional_info:
        message[1].content = message[1].content + '\n' + additional_info
    else:
        pass
    results = db.query(
                      query_texts=[message[1].content],
                      n_results=num_res,
                      where={"section": self.section_name}
                      )
    if additional_info:
        return additional_info + ' '.join(results['documents'][0])
    return ' '.join(results['documents'][0])

  def generate_prompt(self):
    return PDEPPrompt(self.system_prompt, self.user_prompt, context = True).template

  def generate_response(self, db, num_res, rag_context = None):
    if not rag_context:
      rag_context = self.search_rag_res(db, num_res)
    else:
      rag_context = self.search_rag_res(db, num_res, rag_context)
    full_request = {
              **self.user_vars,
              'context': rag_context
              }
    prompt_temp = self.generate_prompt()
    return self.llm(prompt_temp, full_request).content