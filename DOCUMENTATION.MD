# Automated Generation of a German Medical Knowledge Graph Using Large Language Models

## Introduction
The rapid expansion of published medical research presents a significant challenge for healthcare practitioners and clinicians, who must balance demanding daily tasks with staying updated on the latest scientific advancements in their field [^2] [^3]. Open databases like PubMed help address this issue by collecting, storing, and maintaining an ever-growing body of biomedical knowledge. Although healthcare professionals use PubMed to find up-to-date answers to questions arising in their practice, the sheer volume of over 1 million new publications annually makes extracting relevant information time-consuming [^1]. As a result, many questions that cannot be answered within three minutes are often neglected, potentially disadvantaging the affected patients [^4]. This is partly due to the fact that recent knowledge of complex concepts is frequently distributed over several sources of recent research, making the extraction of relevant information and their relationships exceptionally demanding. As a means for centralized aggregation of modular information around entities and their relationships from independent sources, knowledge graphs exhibit the potential to be a crucial pillar for intelligent medical applications to assist healthcare professionals in their day-to-day tasks. This is due to their inherent capability as an information-dense database as well as a foundation for retrieval-augmented question-answering for knowledge-intensive tasks [^5][^6][^7]. While constructing knowledge graphs at scale is a challenging task encompassing highly specialized NLP tasks such as entity detection, linking and graph completion, recent advances in LLMs give reason to assess, whether a certain configuration of interacting LLMs can streamline the construction process with minimal human interference besides quality control after construction [^8]. While the first strides towards such automatic graph construction pipelines in the medical field have been made, no implementation has yet focused on the German language. 
In light of this background, the aim of this project was to examine the possibility of automatically constructing a medical knowledge graph from unstructured German text. For this purpose, a pipeline was constructed applying a combination of GPT-3.5-Turbo and Llama3 Sauerkraut models [^9]. Furthermore, in hindsight of possible application scenarios in retrieval-augmented generation, this project uncovers the possibility of creating a graph-structured vector store, combining the relational information from a knowledge graph with the embedding space and fast information retrieval capabilities of a vector database.

The remainder of this report is structured as follows. We begin with providing a brief overview of previously published work related to the project and highlight our main contributions. Next, we provide a detailed explanation of our methodologies, starting with a general overview and then delving into the logic behind each significant component of the developed pipelines step-by-step. This technical section is followed by a description of the datasets used for information retrieval and evaluation. The fourth section discusses our approach to evaluating and identifying the best possible system with the developed components and analyzes its performance on the described datasets. In the final section, we address the limitations of our proposed system and explore possibilities for future work in both research and open-source software contribution.

## Related Work 

## Methods
In this section, we explain the high-level methodologies employed in this project. Technical details about implementation specifics follow in the subsequent section. 
### Data
#### Wikipedia Documents
For assessing the possibility of constructing a knowledge graph 10.000 Wikipedia pages related to the German word "Krankheit" (="sickness") were extracted. The Wikipedia database identifies each page with a unique page_id, while several versions of the same page might contain different revisions and extensions of Wikipedia contributors over time. For this project, only pages with a unique page_id were retained, resulting in a total dataset of 8.100 pages with a unique page_id. Of these 8.100 pages, all entries were complete except for 10 pages, where no summary of the page was provided. 
For each page, we extracted its full content in "wikitext" format, its title, its page_id, its Wikipedia category tags, its summary description and its outgoing links. As the following procedures are based on the summary, title and content of a page, we focus on these categories in this short dataset description. The following table contains the word count of the extracted page summaries. 

| Wordcount Statistic "Summary" Section | Value      |
|-----------|------------|
| pages used for calculation     | 8090.00 |
| mean      | 72.27  |
| std       | 73.75  |
| min       | 5.00   |
| 25%       | 25.00  |
| 50%       | 43.00  |
| 75%       | 92.00  |
| max       | 718.00 |

As can be seen, the Wikipedia summaries of the pages with a summary are rather short, consisting of 72 words on average. This holds for most pages, as 75% of the summaries contain 92 or less words. As expected for medical texts, the corresponding length is rather high, with 562 characters on average. 

| WordCount Statistic "Full Content" | Value      |
|-----------|------------|
| pages used for calculation     | 8100.00    |
| mean      | 783.78    |
| std       | 1351.20   |
| min       | 26.00     |
| 25%       | 255.00    |
| 50%       | 400.00    |
| 75%       | 744.25    |
| max       | 26051.00  |

The full content of the extracted pages is substantially longer than the summaries. For the 8100 pages, the average word count was 784. While the majority of the data has a wordcount smaller than 744, there exist outliers such as a page with 26.000 words. Thus, chunking text into meaningful sections is going to be a crucial part of data preprocessing. As discussed in the methods section of this report, the semi-parsed wikitext format can be used for this purpose. Special care is taken to avoid cutting off informational dependencies between paragraphs and to maintain the quality of graph construction downstream in the pipeline, especially as the majority of pages only exhibit moderate length.

#### The German Bilingual MeSH
The German Bilingual Medical Subject Headings database (short: MeSH) is a biomedical thesaurus storing English medical terms with their translated German version in a tree-like dependency structure. Originally developed by the U.S. National Library of Medicine, the German MeSH is hosted and maintained by the ZB MED- the German pendant to the U.S. National Library of Medicine - since 2020. The translations and all went through a very careful quality assurance funnel ranging from term identification, term extraction, automatic translation with DeepL to translation curation and translation improvement through subject matter experts. The german MeSH is downloadable in several formats. For this project the JsonLD format was chosen. [^11]
The corresponding file contained 96.745 unique terms consisting of 165.868 words. Some of these terms include undetected duplicates since certain terms are written using varying sentence structures such as "Husten, trockener" and "trockener Husten". However, during evaluation this only increased the likelihood of matching correctly detected nodes in the graph construction procedure, these semantic duplicates were not removed. 


### Pipeline Overview
#### Knowledge Graph Construction Pipeline

## Implementation Details  
### Knowledge Graph Construction Pipeline
#### Data Extraction 
#### Text Preprocessing and Chunking
#### Document Embedding 
#### Knowledge Graph Construction
##### Filtering
##### Prompting
#### Knowledge Graph Storage
#### Knowledge Graph Import and Export

## Evaluation
### Evaluation Set-up
#### Evaluation Data Preparation
#### Graph Evaluation Procedure
### Evaluation Results
#### Quantitative Evaluation Results
#### Qualitative Evaluation Results

## Discussion
### Limitations
### Future Work 
### Conclusion







# References

[^1]: Landhuis, Esther. "Scientific literature: Information overload." Nature 535.7612 (2016): 457-458.

[^2]: Bougioukas, Konstantinos I., et al. "How to keep up to date with medical information using web‐based resources: A systematised review and narrative synthesis." Health Information & Libraries Journal 37.4 (2020): 254-292.

[^3]: Andrews, James E., et al. "Information-seeking behaviors of practitioners in a primary care practice-based research network (PBRN)." Journal of the Medical Library Association 93.2 (2005): 206.

[^4]: Daei, Azra, et al. "Clinical information seeking behavior of physicians: A systematic review." International journal of medical informatics 139 (2020): 104144.

[^5]: Pan, Shirui, et al. "Unifying large language models and knowledge graphs: A roadmap." IEEE Transactions on Knowledge and Data Engineering (2024).

[^6]: Edge, Darren, et al. "From local to global: A graph rag approach to query-focused summarization." arXiv preprint arXiv:2404.16130 (2024).

[^7]: Hogan, Aidan, et al. "Knowledge graphs." ACM Computing Surveys (Csur) 54.4 (2021): 1-37.

[^8]: Min, Bonan, et al. "Recent advances in natural language processing via large pre-trained language models: A survey." ACM Computing Surveys 56.2 (2023): 1-40.

[^9]: VAGOsolutions. (n.d.). Llama-3-SauerkrautLM-8b-Instruct. Hugging Face. Retrieved June 29, 2024, from https://huggingface.co/VAGOsolutions/Llama-3-SauerkrautLM-8b-Instruct

[^10] OpenAI. (2023). GPT-3.5 Turbo. Retrieved June 29, 2024, from https://platform.openai.com/docs/models/gpt-3-5

[^11] "Deutscher MeSH." ZB MED - Informationszentrum Lebenswissenschaften, accessed July 2, 2024, https://www.zbmed.de/open-science/terminologien/deutscher-mesh.
