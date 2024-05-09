from dotenv import load_dotenv
import os
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_core.prompts import ChatPromptTemplate

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
MERGE (s)-[newRelationship:TEIL_VON]->(p)
RETURN count(newRelationship)
"""

page_cat_edge_query="""
MATCH (p:Page)
UNWIND p.categories AS categoryName
MATCH (c:Category {name: categoryName})
MERGE (p)-[r:GEHÖRT_ZU]->(c)
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
MERGE (p)-[r:ERSTE_SECTION]->(s)
RETURN count(r)
"""

sect_auto_nodes_query = """
MATCH (s:Section {section_id: $section_id}), (n {id: $node_id})
CREATE (s)-[r:NENNT]->(n)
RETURN count(r)
"""

# graph transformer prompt
ger_system_prompt = (
"# Anweisungen für die Erstellung eines Wissensgraphen mit GPT-4\n"
"## 1. Überblick\n"
"Du bist ein erstklassiger Algorithmus, der darauf ausgelegt ist, Informationen in strukturierten Formaten zu extrahieren, um einen Wissensgraphen aufzubauen.\n"
"Versuche, so viele Informationen wie möglich aus dem Text zu erfassen, ohne die Genauigkeit zu beeinträchtigen. Füge keine Informationen hinzu, die im Text nicht ausdrücklich erwähnt werden\n"
"- **Knoten** repräsentieren Entitäten und Konzepte.\n"
"- Das Ziel ist es, Einfachheit und Klarheit im Wissensgraphen zu erreichen, um ihn\n"
"für ein breites Publikum zugänglich zu machen.\n"
"## 2. Beschriftung der Knoten\n"
"- **Konsistenz**: Stelle sicher, dass Du verfügbare Typen für Knotenbeschriftungen verwendest.\n"
"Verwende grundlegende oder elementare Typen für Knotenbeschriftungen.\n"
"- Wenn Du beispielsweise eine Entität identifizierst, die eine Person darstellt, "
"bezeichne sie immer als **'Person'**. Vermeide die Verwendung spezifischerer Begriffe "
"wie 'Mathematiker' oder 'Wissenschaftler'"
"  - **Knoten-IDs**: Verwende niemals Ganzzahlen als Knoten-IDs. Knoten-IDs sollten "
"Namen oder menschenlesbare Bezeichnungen sein, die im Text gefunden werden.\n"
"- **Beziehungen** repräsentieren Verbindungen zwischen Entitäten oder Konzepten.\n"
"Stelle bei der Konstruktion von Wissensgraphen Konsistenz und Allgemeinheit in Beziehungstypen sicher."
"Verwende statt spezifischer und momentaner Typen "
"wie 'WURDE_PROFESSOR', allgemeinere und zeitlose Beziehungstypen "
"wie 'PROFESSOR'. Stelle sicher, dass allgemeine und zeitlose Beziehungstypen verwendet werden!\n"
"## 3. Koreferenzauflösung\n"
"- **Beibehaltung der Entitätskonsistenz**: Beim Extrahieren von Entitäten ist es wichtig, "
"sicherzustellen, dass Konsistenz gewahrt wird.\n"
'Wenn eine Entität, wie "John Doe", mehrmals im Text erwähnt wird, '
"aber mit unterschiedlichen Namen oder Pronomen (z. B. 'Joe', 'er') bezeichnet wird, "
"verwende immer die vollständigste Bezeichnung für diese Entität im "
"Wissensgraphen. Verwende in diesem Beispiel 'John Doe' als Entitäts-ID.\n"
"Denke daran, der Wissensgraph sollte kohärent und leicht verständlich sein, "
"so dass die Beibehaltung der Konsistenz in Entitätsverweisen entscheidend ist.\n"
"## 4. Strikte Einhaltung\n"
"Halte Dich strikt an die Regeln. Nichteinhaltung führt zur Beendigung."
)

german_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            ger_system_prompt,
        ),
        (
            "human",
            (
                "Tipp: Stelle sicher, dass Du in dem richtigen Format antwortest."
                "Gebe keinerlei Erklärungen oder Erläuterungen aus. "
                "Sämtliche Bezeichnungen von Knoten und Beziehungen sollen in deutscher Sprache sein."
                "Verwende das angegebene Format,"
                "um Informationen aus der folgenden Eingabe zu extrahieren: {input}"
            ),
        ),
    ]
)



