#Third party packages
import csv


from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from utils.kg_utils import german_prompt

#load env variables
load_dotenv('../config/keys.env', override=True)
NEO4J_URI = os.getenv('NEO4J_URL')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE') or 'neo4j'

#initialize graph transformer and KG
llm = ChatOpenAI(temperature=0, model_name="gpt-4-0125-preview", api_key = OPENAI_API_KEY)
llm_transformer = LLMGraphTransformer(llm=llm, prompt=german_prompt)
kg = Neo4jGraph(
    url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE
)

#create Document objects for pages
with open("../data/small_total_pages.csv") as input_csv, \
     open("../data/skipped_pages_in_graph_transf.csv") as skipped_csv:
    reader = csv.DictReader(input_csv)
    processed_rows = 0
    document_batch = []
    skipped_batch = []
    batch_size = 1
    max_tries = 3

    #create document
    for row in reader:
        processed_rows+=1
        doc = Document(page_content=row["content"])
        document_batch.append(doc)
        if len(document_batch) >= batch_size:
            tries = 0
            while tries < max_tries:
                try:
                    graph_documents = llm_transformer.convert_to_graph_documents(document_batch)
                    print(f"Num Nodes:{len(graph_documents[0].nodes)}")
                    print(f"Num Relationships:{len(graph_documents[0].relationships)}")
                    kg.add_graph_documents(graph_documents=graph_documents)
                    #manually add "mentions" connection
                    document_batch = []
                    tries = 0
                    continue
                except:
                    print("Graph Transformation unsuccessful. Try again")
                    tries +=1

            if tries >= max_tries:
                skipped_batch.append(document_batch)

    if document_batch:
        tries = 0
        while tries < max_tries:
            try:
                graph_documents = llm_transformer.convert_to_graph_documents(document_batch)
                print(f"Num Nodes:{len(graph_documents[0].nodes)}")
                print(f"Num Relationships:{len(graph_documents[0].relationships)}")
                kg.add_graph_documents(graph_documents=graph_documents)
                document_batch = []
                tries = 0
                continue
            except:
                print("Graph Transformation unsuccessful. Try again")
                tries +=1
        if tries >= max_tries:
            skipped_batch.append(document_batch)




print("Finished KG generation")
print(f"Processed: {processed_rows} rows")


