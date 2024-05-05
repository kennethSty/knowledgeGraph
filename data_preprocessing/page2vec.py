#Third party libraries
import torch
import csv

#user libraries
from utils import embed_utils, preprocess_utils

#GerMedBert Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Detected device:", device)
#model = embed_utils.GerMedBert(device)
#print(f"Model on device: {model.device}")
#OpenAI model
model = embed_utils.OpenAIEmbedd()

#outer loop variables for embedding loop
processed_rows = 0
embedded_rows = 0
batch_size = 10
rows_to_embed = []
rows_to_write = []
empty_pages = 0
unique_pages_set = set()

#open files for reading and writing csv output
#note use ../ in local but data in cluster
with open("../data/small_total_pages.csv") as input_csv, \
    open("../data/small_embedded_pages.csv", "w") as output_csv:
    reader = csv.DictReader(input_csv)
    field_names = reader.fieldnames + ['text_to_embed', 'cls_embed']
    writer = csv.DictWriter(output_csv, field_names)
    writer.writeheader() #enforce writing header as otherwise broken csv structure

    for row in reader:
        processed_rows += 1
        # Filter out rows with empty sections, duplicates and of uncompatible data type
        try:
            row['text_to_embed'] = preprocess_utils.get_embedding_text(row=row,keys_to_embed_values=["title", "summary"])
        except ValueError:
            continue
        if row['text_to_embed']=='NA':
            empty_pages +=1
            continue
        if row['text_to_embed'] in unique_pages_set:
            continue

        # Add section to unique set for duplicate check & collect remaining rows for embedding
        unique_pages_set.add(row['text_to_embed'])
        rows_to_embed.append(row)
        # Embed text in batches
        if len(rows_to_embed)>= batch_size:
            text_to_embed_list = [i['text_to_embed'] for i in rows_to_embed]
            batch_embedding = model.embed(text_to_embed_list)
            print(f"Dim of embedded batch: {len(batch_embedding)}")
            print(f"Dim of embedding per seq: {len(batch_embedding[0])}")

            # take batch of embeddings and batch of rows and merge
            for cls_embed,row_to_embed in zip(batch_embedding, rows_to_embed):
                row_to_embed["cls_embed"] = cls_embed
                rows_to_write.append(row_to_embed)
            writer.writerows(rows_to_write)

            # once all processed batch is written, free up the lists again
            embedded_rows += len(rows_to_write)
            print(f"Embedded rows: {embedded_rows}")
            rows_to_embed = []
            rows_to_write = []

    #also embed the last batch
    if rows_to_embed:
        text_to_embed_list = [i['text_to_embed'] for i in rows_to_embed]
        batch_embedding = model.embed(text_to_embed_list)
        print(f"Embedding the last batch of size: {len(batch_embedding)}")

        # take batch of embeddings and batch of rows and merge
        for cls_embed, row_to_embed in zip(batch_embedding, rows_to_embed):
            row_to_embed["cls_embed"] = cls_embed
            rows_to_write.append(row_to_embed)
        writer.writerows(rows_to_write)

        # once all processed batch is written, free up the lists again
        embedded_rows += len(rows_to_write)
        print(f"Finished Embedding")
        print(f"Processed Pages: {processed_rows}")
        print(f"Embedded Pages: {embedded_rows}")
        print(f"Num. unique Pages: {len(unique_pages_set)}")
        print(f"Num. empty Pages: {empty_pages}")

        rows_to_embed = []
        rows_to_write = []




















