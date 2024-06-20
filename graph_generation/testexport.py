from utils.kg_utils import KnowledgeGraph

kg = KnowledgeGraph()
kg.export_to_json("json_Neo4j.json")
kg.query("""
        MATCH (n)
        DETACH DELETE n""")
kg.import_from_json("/Users/Kenneth/PycharmProjects/knowledgeGraph/graph_generation/json_Neo4j.json")

