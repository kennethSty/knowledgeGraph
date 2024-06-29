import ast
import os
from datetime import datetime
from utils import kg_utils
from utils.eval_utils import get_langchain_chroma, get_eval_chain
from langchain_openai import ChatOpenAI


from config import config


def evaluate_graph_transformer(model_name, prompt_strategy, filter_strategy):
    eval_date = datetime.today().strftime('%Y-%m-%d')
    if filter_strategy:
        filter_activation = "filter_on"
    else:
        filter_activation = "filter_off"

    with open("../data/04_eval/mesh_de_total.txt", 'r') as eval_file, \
        open(f"../data/04_eval/{model_name}/{model_name}_{prompt_strategy}_{eval_date}_{filter_activation}_eval_result.txt", 'w') as eval_result, \
        open(f"../data/04_eval/{model_name}/{model_name}_{prompt_strategy}_{eval_date}_{filter_activation}_unmatch_candidates.txt",'w') as unmatch_candidates_txt, \
        open(f"../data/04_eval/{model_name}/{model_name}_{prompt_strategy}_{eval_date}_{filter_activation}_matched_nodes.txt",'w') as matched_node_txt, \
        open(f"../data/04_eval/{model_name}/{model_name}_{prompt_strategy}_{eval_date}_{filter_activation}_unmatched_nodes.txt", 'w') as unmatched_node_txt:
        total_mesh_string = eval_file.read()
        total_mesh = ast.literal_eval(total_mesh_string)
        total_mesh_set = set(total_mesh)

        #get nodes from neo4j
        kg = kg_utils.KnowledgeGraph()
        nodes = kg.query("""
        MATCH (n)
        WHERE NOT 'Chunk' IN labels(n)
          AND NOT 'Section' IN labels(n)
          AND NOT 'Category' IN labels(n)
          AND NOT 'Pages' IN labels(n)
        RETURN n
        """)

        auto_gen_nodes = [dict['n']['id'] for dict in nodes if 'id' in dict['n'].keys()]
        del nodes

        # Calc percentage of terms or heavily related terms in german mesh
        matches = 0
        node_count = 0
        api_call = 0
        unmatched_nodes = []
        matched_nodes = []
        unmatch_candidates = []
        term_matches = 0
        vector_db = get_langchain_chroma()
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        model_eval_chain = get_eval_chain()
        for node in auto_gen_nodes:
            print(f"Processing Node {node_count}/{len(auto_gen_nodes)}")
            node_count+=1
            if (node.lower() in total_mesh_set):
                matches+=1
                term_matches+=1
                matched_nodes.append(node)
            else:
                api_call +=1
                unmatch_candidates.append(node)
                print(f"Check Node with LLM")
                print(f"Number of API calls: {api_call}")
                #if term not directly in mesh, check if there is a synonym or a heavily related term
                similar_mesh_terms = vector_db.similarity_search(node)
                similar_mesh_terms = [term.page_content for term in similar_mesh_terms]
                # let llm decide if the node name is strongly related to at least one of the similar mesh terms
                node_in_mesh = model_eval_chain.invoke(
                    {
                        "input_term": node,
                        "related_terms": str(similar_mesh_terms)
                    }
                )
                if ast.literal_eval(node_in_mesh.content):
                    matches += 1
                    matched_nodes.append(node)
                else:
                    unmatched_nodes.append(node)

        # Write result to files
        if node_count != 0:
            print(f"Percentage of matched auto-generated-nodes: {(matches/node_count)*100} \n")
            eval_result.write(f"Percentage of nodes matched with model check (TP / All): {(matches/node_count)*100} \n")
            eval_result.write(f"Percentage of nodes matched without model check: {term_matches/node_count*100}")
            eval_result.write((f"number of auto-gen nodes: {node_count} \n"))
            eval_result.write(f"number of auto-gen matched nodes with model check: {matches} \n")
            eval_result.write(f"number of auto-gen matched nodes without model check: {term_matches} \n")
            eval_result.write(f"number of nodes exact-term match would have kicked out: {api_call} \n")
            unmatched_node_txt.write(str(unmatched_nodes) + "\n")
            matched_node_txt.write(str(matched_nodes) + "\n")
            unmatch_candidates_txt.write(str(unmatch_candidates) + "\n")
        else:
            print(f"No nodes were auto-generated \n")
            eval_result.write("No nodes were auto-generated \n")
            unmatched_node_txt.write("No nodes were auto-generated \n")
            matched_node_txt.write("No nodes were auto-generated \n")


## Outcomment to run evaluation with latest config settings
"""config = config.load_config()
evaluate_graph_transformer(model_name=config['llm'], prompt_strategy=config['prompt'])"""






