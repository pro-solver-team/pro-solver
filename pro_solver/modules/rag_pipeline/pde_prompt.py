from langchain_core.prompts import ChatPromptTemplate

class PDEPPrompt():
  def __init__(self,
               system_prompt: str,
               user_prompt: str,
               context: bool
               ):
    self.system_prompt = system_prompt
    self.user_prompt = user_prompt
    self.context = context


  @property
  def template(self) -> ChatPromptTemplate:
    if self.context:
        prompt_template = [
            ("system", self.system_prompt),
            ("human", "Use the following context to guide your answer: {context}\n" + self.user_prompt[1])
        ]
    else:
        prompt_template = [
            self.system_prompt,
            self.user_prompt
        ]
    return ChatPromptTemplate.from_messages(prompt_template)