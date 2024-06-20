import csv
import tiktoken
from utils import kg_utils

encoding = tiktoken.get_encoding("cl100k_base")


with open("../data/02_preprocessed/eval_pages_chunked.csv") as page_csv:
    reader = csv.DictReader(page_csv)
    total_tokens = 0
    processed_sections = 0
    for row in reader:
        tokens = encoding.encode(row['section'])
        processed_sections += 1
        total_tokens += len(tokens)
        tokens2 = encoding.encode(kg_utils.med_system_prompt)
        total_tokens += len(tokens2)
    print(f"Total Tokens: {total_tokens}")
    print(f"Avg tokens per Section. {total_tokens/processed_sections}")



