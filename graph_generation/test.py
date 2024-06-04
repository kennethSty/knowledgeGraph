from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.chat_models import ChatOllama

from langchain_experimental.llms.ollama_functions import OllamaFunctions


# Schema for structured response
class Person(BaseModel):
    name: str = Field(description="The person's name", required=True)
    height: float = Field(description="The person's height", required=True)
    hair_color: str = Field(description="The person's hair color")


# Prompt template
prompt = PromptTemplate.from_template(
    """Alex is 5 feet tall. 
Claudia is 1 feet taller than Alex and jumps higher than him. 
Claudia is a brunette and Alex is blonde.

Human: {question}
AI: """
)

# Chain
llm = ChatOllama(model="llama3", temperature=0)
structured_llm = llm.with_structured_output(Person)
chain = prompt | structured_llm