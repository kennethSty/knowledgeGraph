import tiktoken
import ast
from utils.eval_utils import get_langchain_chroma, get_eval_chain

vector_db = get_langchain_chroma()
model_eval_chain = get_eval_chain()
docs = vector_db.similarity_search('Migrationsstörung Von Kortikalen Neuronen')
similar_mesh_terms = [doc.page_content for doc in docs]
print(similar_mesh_terms)
node_in_mesh = model_eval_chain.invoke(
                    {
                        "input_term": 'Migrationsstörung Von Kortikalen Neuronen',
                        "related_terms": str(similar_mesh_terms)
                    }
                )


print(node_in_mesh.content)