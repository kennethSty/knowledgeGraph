import csv
import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")

with open("../data/02_preprocessed/small_chunked_pages.csv") as page_csv:
    reader = csv.DictReader(page_csv)
    total_tokens = 0
    processed_sections = 0
    for row in reader:
        tokens = encoding.encode(row['section'])
        processed_sections += 1
        total_tokens += len(tokens)
    print(f"Total Tokens: {total_tokens}")
    print(f"Avg tokens per Section. {total_tokens/processed_sections}")



