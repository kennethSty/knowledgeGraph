import ast

total_mesh_path = "../data/mesh_de_total.txt"


with open(total_mesh_path, 'r') as eval_file:
    total_mesh_string = eval_file.read()
    total_mesh = ast.literal_eval(total_mesh_string)
    total_mesh_set = set(total_mesh)

    #get nodes from neo4j

    #calc percentage of matches in set




