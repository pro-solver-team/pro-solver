from langchain_mistralai.chat_models import ChatMistralAI

class LLMModel:
    def __init__(self, api_key: str, model_name: str, temperature: float):
        self.model = ChatMistralAI(
                                   model=model_name,
                                   temperature=temperature,
                                   max_retries=2,
                                   api_key=api_key
                                  )

    def __call__(self, prompt_template, input_data: dict):
        llm_chain = prompt_template | self.model
        return llm_chain.invoke(input_data)
