#Third party packages
import pandas as pd
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate

#load env variables
load_dotenv('keys.env', override=True)
NEO4J_URI = os.getenv('NEO4J_URL')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE') or 'neo4j'

#create Document objects for pages
pages = pd.read_csv("../data/pages_until_sroff_9750.csv")
documents = []
for i in [0,1]:
    one_page_text = pages.iloc[i].content[:10000]
    doc = Document(page_content=one_page_text)
    documents.append(doc)

#prepare prompt
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


#test graph transformer on one page
llm = ChatOpenAI(temperature=0, model_name="gpt-4-0125-preview", api_key = OPENAI_API_KEY)
llm_transformer = LLMGraphTransformer(llm=llm, prompt=german_prompt)
graph_documents = llm_transformer.convert_to_graph_documents(documents)
print(f"Num Nodes:{len(graph_documents[0].nodes)}")
print(f"Num Relationships:{len(graph_documents[0].relationships)}")

#loading it into neo4j
kg = Neo4jGraph(
    url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE
)
kg.add_graph_documents(graph_documents=graph_documents)


