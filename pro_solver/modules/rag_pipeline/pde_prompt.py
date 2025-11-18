from langchain_core.prompts import ChatPromptTemplate

class PDEPPrompt():
  def __init__(self,
               system_prompt: tuple,
               user_prompt: tuple = None,
               context: bool = False
               ):
    self.system_prompt = system_prompt
    self.user_prompt = user_prompt
    self.context = context


  def template(self) -> ChatPromptTemplate:
    if not self.context and not self.user_prompt:
        raise ValueError('there is nothing you asked')
    if self.context:
        prompt_template = [
                self.system_prompt,
                ("human",
                 "Use the following context to guide your answer: {context}\n"
                )
                ]
    else:
        prompt_template = [
            self.system_prompt,
            self.user_prompt
        ]
    return ChatPromptTemplate.from_messages(prompt_template)