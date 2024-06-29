# Automatic construction of a german medical knowledge graph using LLMs

## Introduction
The rapid expansion of published medical research presents a significant challenge for healthcare practitioners and clinicians, who must balance demanding daily tasks with staying updated on the latest scientific advancements in their field [^2] [^3]. Open databases like PubMed help address this issue by collecting, storing, and maintaining an ever-growing body of biomedical knowledge. Although healthcare professionals use PubMed to find up-to-date answers to questions arising in their practice, the sheer volume of over 1 million new publications annually makes extracting relevant information time-consuming [^1]. As a result, many questions that cannot be answered within three minutes are often neglected, potentially disadvantaging the affected patients [^4]. This is partly due to the fact that recent knowledge on complex concepts is often distributed over several sources of recent research, making the extraction of relevant information and their relationships exceptionally demanding. As a means for centralized aggregation of modular information around entities and their relationships from independent sources, knowledge graphs exhibits the potential to be a crucial pillar for intelligent medical applications to assist healthcare professionals in their day-to-day tasks. This is due to their inherent capability as an information-dense database as well as a foundation for retrieval-augmented question-answering for knowledge-intensive tasks [^5][^6][^7]. While constructing knowledge graphs at scale is a challenging task encompassing highly specialized NLP tasks such as entity detection, linking and graph completion, recent advanced in LLMs give reason to assess, whether a certain configuration of interacting LLMs can streamline the construction process with minimal human interference besides quality control after construction [^8]. While first strides towards such automatic graph construction pipelines in the medical field have been made, no implementation has yet focused on the german language. 
In light of this background, the aim of this project was to examine the possibility of automatically constructing a medical knowledge graph from unstructured German text. For this purpose a pipeline was constructed applying a combination of GPT-3.5-Turbo and Llama3 Sauerkraut models. Furthermore, in hindsight of possible application scenarios in retrieval-augmented generation, this project uncovers the possibility to create a graph-structured vector store, combining the relational information from a knowledge graph with the embedding space and fast information retrieval capabilities of a vector database.

The remainder of this report is structured as follows. We begin with providing a brief overview of previously published work related to the project, and highlight our main contributions. Next, we provide a detailed explanation of our methodologies, starting with a general overview and then delving into the logic behind each significant component of the developed pipelines step-by-step. This technical section is followed by a description of the datasets used for information retrieval and evaluation. The fourth section discusses our approach to evaluating and identifying the best possible system with the developed components and analyzes its performance on the described datasets. In the final section, we address the limitations of our proposed system and explore possibilities for future work in both research and open-source software contribution.





# References

[^1]: Landhuis, Esther. "Scientific literature: Information overload." Nature 535.7612 (2016): 457-458.

[^2]: Bougioukas, Konstantinos I., et al. "How to keep up to date with medical information using web‐based resources: A systematised review and narrative synthesis." Health Information & Libraries Journal 37.4 (2020): 254-292.

[^3]: Andrews, James E., et al. "Information-seeking behaviors of practitioners in a primary care practice-based research network (PBRN)." Journal of the Medical Library Association 93.2 (2005): 206.

[^4]: Daei, Azra, et al. "Clinical information seeking behavior of physicians: A systematic review." International journal of medical informatics 139 (2020): 104144.

[^5]: Pan, Shirui, et al. "Unifying large language models and knowledge graphs: A roadmap." IEEE Transactions on Knowledge and Data Engineering (2024).

[^6]: Edge, Darren, et al. "From local to global: A graph rag approach to query-focused summarization." arXiv preprint arXiv:2404.16130 (2024).

[^7]: Hogan, Aidan, et al. "Knowledge graphs." ACM Computing Surveys (Csur) 54.4 (2021): 1-37.

[^8]: Min, Bonan, et al. "Recent advances in natural language processing via large pre-trained language models: A survey." ACM Computing Surveys 56.2 (2023): 1-40.
