from dotenv import load_dotenv
import os
import csv
import ast
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_community.graphs.graph_document import Node
from datetime import datetime
import time

#user specified libraries
from utils import kg_utils
from utils.kg_utils import german_prompt, german_med_prompt, llm_checker_prompt
from config import config



def kg_construction(model_name, prompt, framework, until_chunk, prompt_name, filter_node_stragy = True, kg_construction_section_path = "../data/03_model_input/eval_embedded_chunks.csv", kg_construction_page_path = "../data/03_model_input/eval_embedded_pages.csv"):
    """
    Function to generate the knowledge graph
    :param model_name: The model name - only OpenAI based models
    :param prompt: The actual prompt template
    :param framework: The framework used for generating the grpah. At the moment only openai
    :param until_chunk: Until to which chunk of the chunked pages graph data should be extracted. E.g. If the script rea
    :param prompt_name: Name of the prompt used (needed for writing evaluation scripts)
    :param filter_node_stragy: Boolean indicating whether the Llama3 Sauerkraut LLM should be used to filter out nodes not relevant to the specified medical context.
    :param kg_construction_section_path: Path of the chunked pages from whose text the knowledge grpah data should be extracted.
    :param kg_construction_page_path: Path to the pages to which the sections belong in order to create overall page nodes.
    :return: Void - writes result directly into Neo4j

    Note: In order for the function to successfully create a graph the local neo4j server and its DBMS has to be running in the background. This can be achieved by opening the desktop application and "starting" the respective DBMS.
    """

    # Initialize Knowledge Graph & delete existing nodes and relationshiops
    start_time = time.time()
    kg = kg_utils.KnowledgeGraph()
    kg.query("""
        MATCH (n)
        DETACH DELETE n""")

    #get API keys
    load_dotenv('../config/keys.env', override=True)

    #Initialize OpenAI-based Graph Transformer
    if(framework=='openai'):
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        llm = ChatOpenAI(temperature=0, model_name=model_name, api_key = OPENAI_API_KEY) #gpt-4-0125-preview
        llm_transformer = LLMGraphTransformer(llm=llm, prompt=prompt)
        checker_chain = kg_utils.init_llama_chain(model="llama3", prompt=llm_checker_prompt)


    ### GRAPH GENERATION ###
    # Generate Page & Category Nodes
    unique_categories = set()
    with open(kg_construction_page_path) as pages_input:
        pages_reader = csv.DictReader(pages_input)
        processed_rows = 0

        for row in pages_reader:
            #Create page node
            processed_rows+=1
            row["cls_embed"] = ast.literal_eval(row["cls_embed"])
            row["categories"] = ast.literal_eval(row["categories"])
            kg.query(query=kg_utils.merge_page_node_query, params=row)

            #create new category node if not yet existing
            for category in row["categories"]:
                if category not in unique_categories:
                    unique_categories.add(category)
                    cat_dict = {"category_name":category}
                    kg.query(query=kg_utils.merge_category_node_query, params=cat_dict)
                    print(f"Created Category node:{category}")
                    del cat_dict

            #merge-create new page and category nodes
            kg.query(query=kg_utils.merge_page_node_query, params=row)
            #create relationship between all subsequent nodes of a page
            kg.query(query=kg_utils.sect_sect_edge_query, params=row)


        print(f"Created {processed_rows} page-nodes")
        print(f"Created {len(unique_categories)} category-nodes")

    # Sections & Auto Nodes
    eval_date = datetime.today().strftime('%Y-%m-%d')
    with (open(kg_construction_section_path) as input_csv, \
            open(f"../data/04_eval/{model_name}/{model_name}_{prompt_name}_{eval_date}_filter:{filter_node_stragy}_kgconstruction_result.txt", 'w') as eval_log):
        processed_rows = 0
        successful_auto_generations = 0
        reader = csv.DictReader(input_csv)

        #write in new nodes and relationships
        max_tries = 3
        for row in reader:
            #stop creating new nodes after specified chunk
            section_start_time = time.time()
            if processed_rows == until_chunk:
                break
            processed_rows += 1

            #Create Node for each section
            row["cls_embed"] = ast.literal_eval(row["cls_embed"])
            kg.query(query=kg_utils.merge_section_node_query, params = row)

            #automatically extract nodes from each section
            generation_success = False
            tries = 0
            doc = Document(page_content=row["section"])

            #Start max_tries attempts to auto-generate nodes
            print(f"Processing Section: {processed_rows}")
            while (not generation_success) and (tries<max_tries):

                try:
                    graph_documents = llm_transformer.convert_to_graph_documents([doc])

                    if filter_node_stragy:

                        #use llama to filter out medical nodes
                        filtered_nodes = []
                        for node in graph_documents[0].nodes:
                            check_result = checker_chain.invoke({"input": node.id})["text"].strip()
                            if check_result == "True":
                                filtered_nodes.append(Node(id=node.id, type=node.type))

                        #keep only relationships between filtered nodes
                        filtered_relationships = []
                        unique_filtered_nodes = set([node.id for node in filtered_nodes])
                        for relationship in graph_documents[0].relationships:
                            if (relationship.source.id in unique_filtered_nodes and relationship.target.id in unique_filtered_nodes):
                                filtered_relationships.append(relationship)

                        #replace nodes and relationships with filtered versions
                        #do so only if nodes were successfully filtered
                        if(len(filtered_nodes)!=0):

                            with open(f"../data/04_eval/{model_name}/{model_name}_{prompt_name}_{eval_date}_filter:{filter_node_stragy}_nodes_filtered_out.txt",
                            'a') as filtered_out_file:
                                #document node filter result
                                nodes_filtered_out = []
                                for node in graph_documents[0].nodes:
                                    if node.id not in unique_filtered_nodes:
                                        nodes_filtered_out.append(node.id)
                                filtered_out_file.write(f"Section {row['section_id']}: " + str(nodes_filtered_out) + "\n")
                            eval_log.write(f"Node Filter Success \n")
                            eval_log.write(f"Nodes_after_filter / nodes_prev_filter: {len(filtered_nodes)} / {len(graph_documents[0].nodes)} \n")
                            graph_documents[0].nodes = filtered_nodes
                            graph_documents[0].relationships = filtered_relationships
                        else:
                            continue

                    #upload graph documents into neo4j dbms
                    kg.add_graph_documents(graph_documents=list(graph_documents))
                    generation_success = True
                    print(f"Success at try: {tries}")
                except:
                    print(f"No success at try {tries}. Retry")
                    tries+=1

            #If auto-generation  not successfull, continue with next section & log failure
            if not generation_success:
                eval_log.write(f"GENERATION SUCCESS: NO,  Section: {row['section_id']} \n")
                continue
            else:
                section_end_time = time.time()
                proc_time_section = section_end_time - section_start_time
                eval_log.write(f"GENERATION SUCCESS: YES, Section: {row['section_id']} \n")
                eval_log.write(f"Section kg construction took: {proc_time_section} seconds")
                eval_log.write(f"Num. of autom. detected nodes: {len(graph_documents[0].nodes)} \n")
                eval_log.write(f"Num. of autom. detected rel. btw. auto det. nodes: {len(graph_documents[0].relationships)} \n")
                eval_log.write("------------------ \n")
                successful_auto_generations+=1


            #connect auto detected node to each section and corresp. page
            for gdoc in graph_documents:
                for node in gdoc.nodes:
                    section_node_dict = {"section_id": row["section_id"], "node_id": node.id}
                    kg.query(query=kg_utils.sect_auto_nodes_query, params=section_node_dict)

        print(f"Finished: Processed {processed_rows} sections")
        eval_log.write(f"Percent of Sections where auto-generation was successful: {(successful_auto_generations/processed_rows) * 100} \n")
        eval_log.write(f"Total Sections where auto-generation was successful: {successful_auto_generations} \n")
        eval_log.write(f"Finished: Processed {processed_rows} sections \n")

        # Connect sections, pages, categories
        num_sect_page_rels = kg.query(query=kg_utils.sect_page_edge_query)
        num_page_cat_rels = kg.query(query=kg_utils.page_cat_edge_query)
        num_first_rels = kg.query(query=kg_utils.page_first_sect, )
        kg.query(query=kg_utils.sect_sect_edge_query)

        #save the graph as json
        try:
            kg.export_to_json(f"../data/05_graphs/{model_name}/{model_name}_{prompt_name}_{eval_date}_graph.json")
            eval_log.write(f"Graph Exported \n")
            print("Graph Exported")
        except:
            eval_log.write(f"Graph not exported \n")

        #write logging files
        end_time = time.time()
        elapsed_time = end_time-start_time
        eval_log.write(f"Created {num_sect_page_rels} relationships btw. sections and pages \n")
        eval_log.write(f"Created {num_page_cat_rels} relationships btw. pages and categories \n")
        eval_log.write(f"Created {num_first_rels} relationships of pages to their first chunks \n")
        eval_log.write(f"Full KG Construction took {elapsed_time} seconds \n")

    if successful_auto_generations != 0:
        return True
    else:
        return False


















