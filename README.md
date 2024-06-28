# WikiMedGraph
## Contributor
-  [Kenneth Styppa](mailto:kenneth.styppa@web.de) (GitHub alias 'KennyLoRI' and 'Kenneth Styppa')

## Overview
This project utilizes a combination of Langchain, ChromaDB, Neo4j and llama.cpp together with the OpenAI API as well as the Wikipedia API to create a pipeline for automated knowledge-graph construction based on wikipedia articles. The project is structured into modular scripts that each perform a key component of the overall pipeline. These include: 
1. Data extraction from Wikipedia
2. Data Preprocessing to prepare the raw text for knowledge graph construction
3. Knowledge Graph Construction
4. Pipeline Evaluation

## Technologies Used

- **Langchain:** Langchain is a framework for developing applications powered by language models, including information retrievers, text generation pipelines and other wrappers to facilitate a seamless integration of LLM-related open-source software. Within the project Langchain was implemented to build several LLM-chains both with OpenAI models as well as llama.cpp integrations. 

- **ChromaDB:** Chroma DB is an open-source vector storage system designed for efficiently storing and retrieving vector embeddings. In the project ChromaDB was used to store embeddings of the German Mesh vocabulary to facilitate a careful evaluation of the pipelines performance. 

- **llama.cpp:** llama.cpp implements Meta's LLaMa architecture in efficient C/C++ to enable a fast local runtime. Usage of llama.cpp is facilitate by a langchain wrapper, making it possible to run quantized versions of llama based architectures on a local GPU. 

- **OpenAI API:** The OpenAI API provides accesss to OpenAIs language models, enabling integration of highly performant LLM capabilities into programatic applications. Within the scope of the project the OpenAI API was used for two purposes: 1) To generate embeddings for the extracted wikipedia text. 2) To automatically extract knowledge-graph data out of wikipedia sections based on Neo4j's GraphTransformer implementation in langchain_community. 

- **Llama3 Sauerkraut:** Llama3 Sauerkraut is an advanced implementation of the LLaMa3 architecture, fine-tuned and specialized for the German language. Llama3 Sauerkraut was used in the project within a second LLM pipeline, which takes the autoamtically detected graph data from the OpenAI model as an input and filters out the nodes relevant for the provided context. 

- **Neo4j:** Neo4j Graph Database: Neo4j is a highly scalable, native graph database. In the project, Neo4j was used as the database to store and grow the generated detected relational data in a knowledge graph. For this purpose a local Neo4j with apoc plugin as well as read and write file access was initialized.

- **Wikipedia API:** To extract the data for this project, germamn wikipedia pages from the context of "sickness" were extracted via the Wikipedia API. The basis of the API extraction was based on the wikipedia python package, which was overloaded and extended in certain functionalities key for extracting the data.

- 1. **Prerequisites:**
   - Ensure you have Python installed on your system. Your Python version should match 3.10.
   - Ensure to have conda installed on your system.
   - Create a folder where you want to store the project.

2. **Create a Conda Environment:**
   - Create a conda environment
   - Activate the environment
   ```bash
   conda create --name your_project_env python=3.10
   conda activate your_project_env
   ```

4. **Clone the Repository into your working directory:**
   ```bash
   git clone git@github.com:bgzdaniel/PubMedTempGraph.git
   ```
   When using Mac set pgk_config path:
   ```bash
   export PKG_CONFIG_PATH="/opt/homebrew/opt/openblas/lib/pkgconfig"
   ```

   then switch to the working directory of the project:
   ```bash
   cd PubMedTempGraph
   ```
   
6. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
7. **Llama.cpp GPU installation:**
   (When using CPU only, skip this step.)

   This part might be slightly tricky, depending on which system the installation is done. We do NOT recommend installation on Windows. It has been tested, but requires multiple components which need to be downloaded. Please contact [Daniel Bogacz](mailto:daniel.bogacz@stud.uni-heidelberg.de) for details.

   **Linux:**
   ```bash
   CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
   ```

   **MacOS:**
   ```bash
   CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
   ```

If anything goes wrong in this step, please contact [Kenneth Styppa](mailto:kenneth.styppa@web.de) for **MacOS** installation issues. Also refer to the installation guide provided [here](https://python.langchain.com/docs/integrations/llms/llamacpp) and also [here](https://llama-cpp-python.readthedocs.io/en/latest/install/macos/)

6. **Data & Model Set-up:**
   - Download the [model file](https://huggingface.co/VAGOsolutions/Llama-3-SauerkrautLM-8b-Instruct). Insert the model file at `models/Llama-3-SauerkrautLM-8b-Instruct-Q5_K_M.gguf`.
   - For a minimal test setup go to [this](https://drive.google.com/drive/folders/1-8hVX75ui3wtk-4OPQckqe8EdvQtv-1H?usp=sharing) google drive link and download the data folder (folder called `data`). Insert the data folder store in the `data` directory.
   - Set up an OpenAI Account [here](https://openai.com/index/openai-api/).
   - Download Neo4j [here](https://neo4j.com/download/) and follow the installation guide [here](https://neo4j.com/docs/operations-manual/current/installation/osx/). My implementation was configured with OpenJDK 17. After completing the installation, open Neo4j on your local machine and set up the apoc plugin. Also open the directory in your NEO4J HOME where the neo4j.config is located and create a apoc.config file. In this file include 'apoc.import.file.use_neo4j_config=false'. Note this allows Neo4j to read any file on your system, but for importing the Knowledge Graph from a json file on your machine this is the prefered way. For other way of importing knowledge graph data please refer to the advanced apoc documentation of Neo4j [here](https://neo4j.com/labs/apoc/4.2/overview/apoc.import/).
   - Place your API keys and your Neo4j database credentials in config/keys.env in the following format: 
  ```bash
  NEO4J_URL=bolt://localhost:7687
  NEO4J_USERNAME=
  NEO4J_PASSWORD=
  OPENAI_API_KEY=
   ```






