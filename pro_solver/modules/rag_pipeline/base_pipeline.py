from langchain_core.prompts import ChatPromptTemplate
from pro_solver.modules.rag_pipeline.pde_prompt import PDEPPrompt
from pro_solver.modules.rag_pipeline.base_model import LLMModel

class ModelPipeline():
  def __init__(self,
               model: LLMModel,
               rag_prompt: str,
               rag_vars: dict,
               user_prompt: str,
               user_vars: dict,
               system_prompt: str,
               section_name: str
               ):
    self.llm = model
    self.rag_temp = ChatPromptTemplate.from_messages(rag_prompt)
    self.rag_vars = rag_vars
    self.system_prompt = system_prompt
    self.user_vars = user_vars
    self.user_prompt = user_prompt
    self.section_name = section_name

  def search_rag_res(self, db, num_res):
    message = self.rag_temp.format_messages(**self.rag_vars)
    results = db.query(
                      query_texts=[message[1].content],
                      n_results=num_res,
                      where={"section": self.section_name}
                      )
    return ' '.join(results['documents'][0])

  def generate_prompt(self):
    return PDEPPrompt(self.system_prompt, self.user_prompt, context = True).template

  def generate_response(self, db, num_res, rag_context = None):
    if not rag_context:
      rag_context = self.search_rag_res(db, num_res)
    else:
      pass
    full_request = {
              **self.user_vars,
              'context': rag_context
              }
    prompt_temp = self.generate_prompt()
    return self.llm(prompt_temp, full_request).content