from graph_generation.kg_construction import kg_construction
from config import config
from utils.kg_utils import german_prompt, german_med_prompt

#load model settings from parameters.yml
config = config.load_config()
if config['prompt']=='german_med_prompt':
    prompt = german_med_prompt
else:
    prompt = german_prompt

print(f"Start Construction with model:{config['llm']}, {config['prompt']} and filter={config['filter_node_strategy']}")
construction_success = kg_construction.kg_construction(model_name=config['llm'],
                                prompt=prompt,
                                framework=config['llm_framework'],
                                until_chunk=config['until_chunk'],
                                prompt_name=config['prompt'],
                                filter_node_stragy=config['filter_node_strategy'],
                                kg_construction_section_path =config['kg_construction_section_path'],
                                kg_construction_page_path = config['kg_construction_page_path'])

print(f"Construction Success: {construction_success}")
