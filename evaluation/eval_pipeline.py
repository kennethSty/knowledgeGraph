from mesh_evaluation import evaluate_graph_transformer
from graph_generation import kg_construction
from utils.kg_utils import german_prompt, german_med_prompt
from config import config

#keep general default settings constant
config = config.load_config()

# Create evaluation settings mesh
eval_settings = []
models = ['gpt-4', 'gpt-4o', 'gpt-3.5-turbo']
prompt_strategies = ['german_prompt', 'german_med_prompt']
for model in models:
    for prompt in prompt_strategies:
        eval_settings.append({'model_name': model, 'prompt': prompt})

# Execute kg construction for each eval combination & evaluate immediately afterwards
for eval_comb in eval_settings:
    if eval_comb['prompt'] == 'german_prompt':
        prompt = german_prompt
    if eval_comb['prompt'] == 'german_med_prompt':
        prompt = german_med_prompt

    print(f"Start Construction with model:{eval_comb['model_name']} and {eval_comb['prompt']}")
    construction_success = kg_construction.kg_construction(model_name=eval_comb['model_name'],
                                    prompt=prompt,
                                    framework=config['llm_framework'],
                                    until_chunk=config['until_chunk'],
                                    prompt_name=eval_comb['prompt'])
    if construction_success:
        print(f"Start Evaluation with model: {eval_comb['model_name']} and {eval_comb['prompt']}")
        evaluate_graph_transformer(model_name=eval_comb['model_name'],
                                   prompt_strategy=eval_comb['prompt'])
    else:
        continue






