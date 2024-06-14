from utils import kg_utils
from utils.kg_utils import  llm_checker_prompt
checker_chain = kg_utils.init_llama_chain(model="llama3", prompt=llm_checker_prompt)

result = checker_chain.invoke({"input": "Dosis"})
print(result)

