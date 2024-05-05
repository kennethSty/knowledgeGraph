from dotenv import load_dotenv
import os
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector

class KnowledgeGraph(Neo4jGraph):
    # load env variables
    def __init__(self):
        load_dotenv('keys.env', override=True)
        self.NEO4J_URI = os.getenv('NEO4J_URL')
        self.NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
        self.NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
        self.NEO4J_DATABASE = os.getenv('NEO4J_DATABASE') or 'neo4j'

        super().__init__(
            url=self.NEO4J_URI,
            username=self.NEO4J_USERNAME,
            password=self.NEO4J_PASSWORD,
            database=self.NEO4J_DATABASE
        )

        self.query("""
        CREATE CONSTRAINT unique_section IF NOT EXISTS 
            FOR (s:Section) REQUIRE s.section_id IS UNIQUE
        """)

        print(f"Graph database initialized: {self.NEO4J_DATABASE}")



# cypher statements for creating nodes
merge_section_node_query = """
MERGE(s:Section {section_id: $section_id})
    ON CREATE SET 
        s.section_title = $section_title, 
        s.text_to_embed = $text_to_embed, 
        s.page_id = $page_id,
        s.cls_embed = $cls_embed, 
        s.section_counter = $section_counter
RETURN s
"""

merge_page_node_query = """
MERGE(p:Page {page_id: $page_id})
    ON CREATE SET 
        p.page_title = $title,
        p.text_to_embed = $text_to_embed, 
        p.cls_embed = $cls_embed, 
        p.section_ids = $section_ids,
        p.categories = $categories
RETURN p
"""

merge_category_node_query = """
MERGE(cat:Category {name: $category_name})
RETURN cat
"""

# cypher statements for creating relationships
sect_page_edge_query = """
MATCH (p:Page), (s:Section)
WHERE p.page_id = s.page_id
MERGE (s)-[newRelationship:PART_OF]->(p)
RETURN count(newRelationship)
"""

page_cat_edge_query="""
MATCH (p:Page)
UNWIND p.categories AS categoryName
MATCH (c:Category {name: categoryName})
MERGE (p)-[r:BELONGS_TO]->(c)
RETURN count(r)
"""

# creates connection "NEXT" between subsequent chunks (like linked list)
sect_sect_edge_query="""
  MATCH (from_same_page:Section)
  WHERE from_same_page.page_id = $page_id
  WITH from_same_page
    ORDER BY from_same_page.section_counter ASC
  WITH collect(from_same_page) as section_list
    CALL apoc.nodes.link(
        section_list, 
        "NEXT", 
        {avoidDuplicates: true}
    )
  RETURN size(section_list)
"""

page_first_sect = """
MATCH (p:Page), (s:Section)
WHERE p.page_id = s.page_id
    AND s.section_counter = "0"
MERGE (p)-[r:FIRST_SECTION]->(s)
RETURN count(r)
"""




