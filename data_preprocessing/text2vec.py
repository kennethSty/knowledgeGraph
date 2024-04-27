#Third party libraries
import torch
import csv

#user libraries
from utils import embed_utils, preprocess_utils

#getting model ready for embed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = embed_utils.GerMedBert(device)


with open("../data/chunked_pages.csv") as input_csv, \
    open("../data/embedded_chunks.csv", "w") as output_csv:
    reader = csv.DictReader(input_csv)
    writer = csv.DictWriter(output_csv, fieldnames=["page_title", "text_to_embed", "page_id", "links", "categories", "summary", "subsection_ids"])

    batch_size = 50
    rows_to_embed = []

    for row in reader:
        row['text_to_embed'] = preprocess_utils.get_embedding_text(row=row, keys_to_embed_values=["page_title", "section_title", "section"])
        rows_to_embed.append(row)
        if len(rows_to_embed)>= batch_size:
            cls_embed_batch = model.embed(rows_to_embed)

            #todo add find good datastructure for embeddings
            #iterate over embedding, metadate etc always creating a nice dict of embed, und meta info. ready für knoten.














