import pickle

from dotenv import load_dotenv
import os
import csv
import ast
import torch
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from transformers import pipeline
from langchain_community.chat_models import ChatOllama
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_community.chat_models import ChatOllama
from utils.structured_llm_class import StructuredLanguageModel

import sys



#user specified libraries
#sys.path.extend(['/pfs/data5/home/hd/hd_hd/hd_aa304/utils', '/pfs/data5/home/hd/hd_hd/hd_aa304/config']) #only cluster
from utils import kg_utils
from utils.kg_utils import german_prompt
from config import config

#get llm settings from config
load_dotenv('../../config/keys.env', override=True)
config = config.load_config()

#OpenAI Models
if(config['llm_framework']=='openai'):
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    llm = ChatOpenAI(temperature=0, model_name=config['llm'], api_key = OPENAI_API_KEY) #gpt-4-0125-preview

#LLama models
if(config['llm']=='llama3'):
    if(config['modelling_location']=='local' and config['llm_framework']=="ollama"):
        llm = StructuredLanguageModel(model="llama3")
        llm_transformer = LLMGraphTransformer(llm=llm, prompt=german_prompt)
    elif(config['modelling_location']=='local' and config['llm_framework']=="llama.cpp"):
        llm = kg_utils.instantiate_llm()
        llm_transformer = kg_utils.LLamaGraphTransformer(llm = llm, prompt=german_prompt)

    elif (config['modelling_location'] == 'cluster'):
        #set device etc if device CPU break
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Detected device:", device)
        llm = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B", device_map='auto')
        llm_transformer =  kg_utils.LLamaGraphTransformer(llm = llm, prompt=german_prompt)

if (config['llm'] == 'llama2' and config['local_modelling']):
    if (config['modelling_location'] == 'local'):
        llm = kg_utils.instantiate_llm()
        llm_transformer = kg_utils.LLamaGraphTransformer(llm=llm, prompt=german_prompt)
    elif (config['modelling_location'] == 'cluster'):
        llm = pipeline("text-generation", model="meta-llama/Llama-2-7b-hf", device_map='auto')
        llm_transformer = kg_utils.LLamaGraphTransformer(llm=llm, prompt=german_prompt)


# Sections & Auto Nodes
generated_graph_dict = {}
#note without ../ only on cluster. other ../data_cluster
with open("../../data_cluster/small_embedded_chunks.csv") as input_csv, \
     open("../../data_cluster/node_list.csv", "w") as nodes_csv, \
     open("../../data_cluster/edge_list.csv", "w") as edges_csv:


        #prepare files for reading and writing
        reader = csv.DictReader(input_csv)
        node_field_names = ['section_id', 'id', 'type']
        edge_field_names = ['source_id', 'source_type', 'target_id', 'target_type', 'edge_type']
        node_writer = csv.DictWriter(nodes_csv, node_field_names)
        edge_writer = csv.DictWriter(edges_csv, edge_field_names)
        node_writer.writeheader()
        edge_writer.writeheader()
        nodes_to_write = []
        edges_to_write = []
        batch_size = 10 #write every 10 rows in batch
        processed_rows = 0
        generated_nodes = 0
        generated_edges = 0

        #write in new nodes and relationships
        for row in reader:
            if processed_rows == 1:
                break ##only for test
            processed_rows+=1
            print(f"Processing section number: {processed_rows}")

            #automatically extract nodes & relationships from each section
            doc = Document(page_content=row["section"])
            graph_documents = llm_transformer.convert_to_graph_documents([doc])

            for gdoc in graph_documents:
                # collect nodes to write
                for node in gdoc.nodes:
                    node_dict = dict(node)
                    del node_dict['properties']
                    node_dict["section_id"]=row["section_id"]
                    nodes_to_write.append(node_dict)

                # write nodes
                node_writer.writerows(nodes_to_write)
                generated_nodes += len(nodes_to_write)
                nodes_to_write = []

                #collect edges to write
                for edge in gdoc.relationships:
                    edge_dict = {"source_id": edge.source.id,
                                 "source_type": edge.source.type,
                                 "target_id": edge.target.id,
                                 "target_type": edge.target.type,
                                 "edge_type": edge.type,
                                 }
                    edges_to_write.append(edge_dict)

                #write edges
                edge_writer.writerows(edges_to_write)
                generated_edges += len(edges_to_write)
                edges_to_write = []


        print(f"Finished: Processed {processed_rows} sections")
        print(f"Finished: Generated {generated_nodes} nodes")
        print(f"Finished: Generated {generated_edges} edges")











