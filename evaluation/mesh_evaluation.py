import ast
from utils import kg_utils

with open("../data/04_eval/mesh_de_total.txt", 'r') as eval_file, \
    open("../data/04_eval/eval_result.txt", 'w') as eval_result, \
    open("../data/04_eval/unmatched_nodes.txt", 'w') as unmatched_node_txt:
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
    unmatched_nodes = []
    for node in auto_gen_nodes:
        if (node.lower() in total_mesh_set):
            matches+=1
        else:
            unmatched_nodes.append(node)


    # Write result to files
    print(f"Percentage of matched auto-generated-nodes: {(matches/len(auto_gen_nodes))*100} ")
    eval_result.write(f"Percentage of matched auto-generated-nodes: {(matches/len(auto_gen_nodes))*100} ")
    unmatched_node_txt.write(str(unmatched_nodes))






