import torch
import csv
import spacy
from scipy.signal import argrelextrema
import numpy as np

# user libraries
from utils import preprocess_utils, extract_utils

#enable processing on large csv fiiles
preprocess_utils.increase_csv_maxsize()

with open("../data/pages_until_sroff_9750.csv") as input_csv, open("../data/chunked_pages.csv", "w") as output_csv:
    reader = csv.DictReader(input_csv)
    writer = csv.DictWriter(output_csv, fieldnames=["title","pageid","content","links","categories","head_sections","summary", "text_chunks"])

    paragraphs_rows = []
    batch_size = 5
    processed = 0
    max_tokens = 0
    for row in reader:
        sections, head_sections = extract_utils.extract_sections(row["content"])
        row["head_sections"] = str(head_sections)
        row["text_chunks"] = str(sections)
        del row["content"]
        paragraphs_rows.append(row)
        if len(paragraphs_rows) >= batch_size:
            writer.writerows(paragraphs_rows) #TODO solve bug that it cannot write back
            processed += len(paragraphs_rows)
            paragraphs_rows = []
            print(f"until now processed {processed} documents")

        #also write out last batch if batch_size not reached
    if paragraphs_rows:
        writer.writerows(paragraphs_rows)
        processed += len(paragraphs_rows)
        paragraphs_rows = []

    print(f"processed in total {processed} documents")
