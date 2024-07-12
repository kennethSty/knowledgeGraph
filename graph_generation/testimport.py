from utils.kg_utils import KnowledgeGraph

kg = KnowledgeGraph()
#kg.export_to_json("json_Neo4j.json")
kg.query("""
        MATCH (n)
        DETACH DELETE n""")
kg.import_from_json(".../data/05_graphs/gpt-3.5-turbo/gpt-3.5-turbo_german_med_prompt_2024-06-25_graph.json")

