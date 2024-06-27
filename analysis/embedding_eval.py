import tiktoken
import ast
from utils.eval_utils import get_langchain_chroma, get_eval_chain

vector_db = get_langchain_chroma()
model_eval_chain = get_eval_chain()
docs = vector_db.similarity_search('Beta-Isomere')
similar_mesh_terms = [doc.page_content for doc in docs]
print(similar_mesh_terms)
node_in_mesh = model_eval_chain.invoke(
                    {
                        "input_term": 'Beta-Isomere',
                        "related_terms": str(similar_mesh_terms)
                    }
                )


print(node_in_mesh.content)