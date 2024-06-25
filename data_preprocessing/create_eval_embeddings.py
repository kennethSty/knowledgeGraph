from langchain_chroma import Chroma
from utils.eval_utils import get_langchain_chroma
import chromadb
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import ast

load_dotenv('../config/keys.env', override=True)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
with open("../data/04_eval/mesh_de_total.txt", "r") as input_file:
    mesh = ast.literal_eval(input_file.read())

#Create Vectorstore - first delete old to ensure clear input
clear_collection = get_langchain_chroma()
clear_collection.delete_collection()
vector_db = get_langchain_chroma()
load_dotenv('../config/keys.env', override=True)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)

term_batch = []
processed_terms = 0
for term in list(mesh):
    term_batch.append(term)
    processed_terms += 1
    if len(term_batch) > 50:
        #Chroma.from_texts(collection_name = "mesh_embeddings", texts=list(term_batch), embedding=embeddings, persist_directory="/Users/Kenneth/PycharmProjects/knowledgeGraph/data/04_eval/chroma_store")
        vector_db.add_texts(term_batch)
        term_batch = []
        print(f"Embedded {processed_terms} terms")

if term_batch:
    #Chroma.from_texts(collection_name = "mesh_embeddings", texts=list(term_batch), embedding=embeddings,persist_directory="/Users/Kenneth/PycharmProjects/knowledgeGraph/data/04_eval/chroma_store")
    vector_db.add_texts(term_batch)
    term_batch = []

numb_docs = vector_db._collection.count()
print(f"Created {numb_docs} embeddings")