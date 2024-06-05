import ast
from datetime import datetime
from utils import kg_utils
from config import config


def evaluate_graph_transformer(model_name, prompt_strategy):
    eval_date = datetime.today().strftime('%Y-%m-%d')
    with open("../data/04_eval/mesh_de_total.txt", 'r') as eval_file, \
        open(f"../data/04_eval/{model_name}/{model_name}_{prompt_strategy}_{eval_date}_eval_result.txt", 'w') as eval_result, \
        open(f"../data/04_eval/{model_name}/{model_name}_{prompt_strategy}_{eval_date}_matched_nodes.txt",'w') as matched_node_txt, \
        open(f"../data/04_eval/{model_name}/{model_name}_{prompt_strategy}_{eval_date}_unmatched_nodes.txt", 'w') as unmatched_node_txt:
        total_mesh_string = eval_file.read()
        total_mesh = ast.literal_eval(total_mesh_string)
        total_mesh_set = set(total_mesh)

        #get nodes from neo4j
        kg = kg_utils.KnowledgeGraph()
        nodes = kg.query("""
            MATCH (n)
            RETURN n""")

        auto_gen_nodes = [dict['n']['id'] for dict in nodes if 'id' in dict['n'].keys()]
        del nodes

        # Calc percentage of terms in german mesh
        matches = 0
        node_count = 0
        unmatched_nodes = []
        matched_nodes = []
        for node in auto_gen_nodes:
            node_count+=1
            if (node.lower() in total_mesh_set):
                matches+=1
                matched_nodes.append(node)
            else:
                unmatched_nodes.append(node)


        # Write result to files
        if node_count != 0:
            print(f"Percentage of matched auto-generated-nodes: {(matches/node_count)*100} ")
            eval_result.write(f"Percentage of matched auto-generated-nodes: {(matches/node_count)*100} \n")
            eval_result.write((f"number of auto-gen nodes: {node_count}"))
            eval_result.write((f"number of auto-gen matched nodes: {matches}"))
            unmatched_node_txt.write(str(unmatched_nodes))
            matched_node_txt.write(str(matched_nodes))
        else:
            print(f"No nodes were auto-generated")
            eval_result.write("No nodes were auto-generated")
            unmatched_node_txt.write("No nodes were auto-generated")
            matched_node_txt.write("No nodes were auto-generated")


## Outcomment to run evaluation with latest config settings
"""config = config.load_config()
evaluate_graph_transformer(model_name=config['llm'], prompt_strategy=config['prompt'])"""






