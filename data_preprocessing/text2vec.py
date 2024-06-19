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
model = embed_utils.OpenAIEmbedd(model_name = "text-embedding-3-small")

#outer loop variables for embedding loop
processed_rows = 0
embedded_rows = 0
batch_size = 8
rows_to_embed = []
rows_to_write = []
empty_sections = 0
unique_section_set = set()

#open files for reading and writing csv output
#note use ../ in local but data in cluster
with open("../data/02_preprocessed/eval_pages_chunked.csv") as input_csv, \
    open("../data/03_model_input/eval_embedded_chunks.csv", "w") as output_csv:
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
        if row['section_id'] in unique_section_set:
            continue
            print("duplicate skipped")

        # Add section to unique set for duplicate check & collect remaining rows for embedding
        unique_section_set.add(row['section_id'])
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
            # once all processed batch is written, free up the lists again
            writer.writerows(rows_to_write)
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
        print(f"Processed Sections: {processed_rows}")
        print(f"Embedded sections: {embedded_rows}")
        print(f"Num. unique sections: {len(unique_section_set)}")
        print(f"Num. empty sections: {empty_sections}")

        rows_to_embed = []
        rows_to_write = []




















