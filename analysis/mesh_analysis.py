import ast

with open("../data/04_eval/mesh_de_total.txt") as input:
    mesh = input.read()
    number_terms = len(ast.literal_eval(mesh))
    word_count = 0
    for term in ast.literal_eval(mesh):
        word_count += len(term.split())

    print(f"Number of words in Mesh: {word_count}")
    print(f"Number of terms in Mesh: {number_terms}")

