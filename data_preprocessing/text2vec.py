#Third party libraries
import torch
import csv

#user libraries
from utils import embed_utils, preprocess_utils

#getting model ready for embed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Detected device:", device)
model = embed_utils.GerMedBert(device)
print(f"Model on device: {model.device}")

#outer loop variables for embedding loop
processed_rows = 0
embedded_rows = 0
batch_size = 2
rows_to_embed = []
rows_to_write = []
empty_sections = 0
unique_section_set = set()

#open files for reading and writing csv output
#note use ../ in local but data in cluster
with open("../data/chunked_pages.csv") as input_csv, \
    open("../data/embedded_chunks.csv", "w") as output_csv:
    reader = csv.DictReader(input_csv)
    field_names = reader.fieldnames + ['text_to_embed', 'cls_embed']
    writer = csv.DictWriter(output_csv, field_names)
    writer.writeheader() #enforce writing header as otherwise broken csv structure

    for row in reader:
        processed_rows += 1
        # Filter out rows with empty sections, duplicates and of uncompatible data type
        try:
            row['text_to_embed'] = preprocess_utils.get_embedding_text(row=row,                                                            keys_to_embed_values=["page_title", "section_title", "section"])
        except ValueError:
            continue
        if row['section']=='NA':
            empty_sections +=1
            continue
        if row['section'] in unique_section_set:
            continue

        # Add section to unique set for duplicate check & collect remaining rows for embedding
        unique_section_set.add(row['section'])
        rows_to_embed.append(row)
        # Embed text in batches
        if len(rows_to_embed)>= batch_size:
            text_to_embed_list = [i['text_to_embed'] for i in rows_to_embed]
            cls_embed_batch = model.embed(text_to_embed_list)
            print(f"Dim of embedded batch (batchsize, emb_dim): {cls_embed_batch.size()}")

            # take batch of embeddings and batch of rows and merge
            for cls_embed,row_to_embed in zip(cls_embed_batch, rows_to_embed):
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
        cls_embed_batch = model.embed(text_to_embed_list)
        print(f"Embedding the last batch of size: {len(cls_embed_batch)}")

        # take batch of embeddings and batch of rows and merge
        for cls_embed, row_to_embed in zip(cls_embed_batch, rows_to_embed):
            row_to_embed["cls_embed"] = cls_embed
            rows_to_write.append(row_to_embed)
            writer.writerows(rows_to_write)

        # once all processed batch is written, free up the lists again
        embedded_rows += len(rows_to_write)
        print(f"Finished Embedding")
        print(f"Processed Sections: {processed_rows}")
        print(f"Embedded sections: {embedded_rows}")
        print(f"Num. unique sections: {len(unique_section_set)}")
        print(f"Num. empty sections: {empty_sections}")

        rows_to_embed = []
        rows_to_write = []




















