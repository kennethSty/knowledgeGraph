import csv
import tiktoken
from utils import kg_utils

encoding = tiktoken.get_encoding("cl100k_base")
input_cost_per_mill = 0.5
output_cost_per_mill = 1.5

with open("../data/04_eval/selected_eval_embedded_chunks.csv") as chunk_csv:
    reader = csv.DictReader(chunk_csv)
    total_tokens = 0
    processed_sections = 0
    for row in reader:
        tokens = encoding.encode(row['section'])
        processed_sections += 1
        total_tokens += len(tokens)
        tokens_system_prompt = encoding.encode(kg_utils.german_med_prompt.messages[0].prompt.template)
        total_tokens += len(tokens_system_prompt)
        tokens_human_prompt = encoding.encode(kg_utils.german_med_prompt.messages[1].prompt.template)
        total_tokens += len(tokens_human_prompt)

    print(f"Total Tokens: {total_tokens}")
    print(f"Number of Sections: {processed_sections}")
    print(f"Cost per input token: {input_cost_per_mill}$ per 1M tokens")
    print(f"Cost per output token: {output_cost_per_mill}$ per 1M tokens")
    print(f"Total input cost: {round((total_tokens*input_cost_per_mill)/1000000, 2)}")
    print(f"Estimated output length in worst-case same as input length")
    print(f"Estimated input + output cost: {round((total_tokens*input_cost_per_mill)/1000000, 4) + round((total_tokens * output_cost_per_mill)/1000000, 4)}")
    print(f"Avg tokens per Section. {total_tokens/processed_sections}")



