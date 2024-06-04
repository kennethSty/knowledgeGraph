import ast
import csv
import spacy

# user libraries
from utils import preprocess_utils


#enable processing on large csv fiiles
preprocess_utils.increase_csv_maxsize()
nlp = spacy.load("de_core_news_sm")

#Note: cluster its data/somepath and for local it is ../data/somepath
with open("../data/00_raw/pages_until_sroff_10.csv") as input_csv, \
        open("../data/02_preprocessed/small_chunked_pages.csv", "w") as paragraph_output_csv, \
        open("../data/02_preprocessed/small_total_pages.csv", "w") as page_output_csv:
    reader = csv.DictReader(input_csv)

    fieldnames_paragraph_writer = ["page_title", "page_id", "section", "section_title", "section_id", "section_counter"]
    fieldnames_page_writer = ["title", "page_id", "links", "categories", "summary", "section_ids", "content"]

    paragraph_writer = csv.DictWriter(paragraph_output_csv, fieldnames=fieldnames_paragraph_writer)
    page_writer = csv.DictWriter(page_output_csv, fieldnames=fieldnames_page_writer)
    paragraph_writer.writeheader()
    page_writer.writeheader()

    paragraphs_rows = []
    page_rows = []
    batch_size = 1
    processed = 0
    max_tokens = 0
    empty_paragraphs = 0
    tokens_sum = 0
    processed_pages = 0
    for row in reader:
        #extract sections
        sections, head_section_titles = preprocess_utils.extract_sections(row["content"])
        row["content"]

        #rename and delete to free memory
        row["page_id"] = row["pageid"]
        del row["head_sections"], row["pageid"]

        #write sections into separate file
        for i in range(len(head_section_titles)):
            paragraph_row = {"section": sections[i],
                             "section_title": head_section_titles[i],
                             "page_title": row["title"],
                             "page_id": row["page_id"],
                             "section_counter": f"{i}",
                             "section_id": f"{i}-{row['page_id']}"}
            if len(paragraph_row["section"]) == 0:
                empty_paragraphs+=1
                print("Empty paragraph skipped.")
            else:
                paragraphs_rows.append(paragraph_row)
                tokens_in_paragraph = nlp(paragraph_row["section"])
                tokens_sum+=len(tokens_in_paragraph)
        if len(paragraphs_rows) >= batch_size:
            paragraph_writer.writerows(paragraphs_rows)
            processed += len(paragraphs_rows)
            paragraphs_rows = []
            print(f"until now processed {processed} paragraphs")

        #write pages into separate file
        row["categories"] = str([preprocess_utils.extract_category(category) for category in ast.literal_eval(row["categories"])])
        page_rows.append(row)
        if len(page_rows)>= batch_size:
            page_writer.writerows(page_rows)
            processed_pages += len(page_rows)
            page_rows = []
            print(f"until now processed {processed_pages} pages")

    #also write out last batch if batch_size not reached
    if paragraphs_rows:
        paragraph_writer.writerows(paragraphs_rows)
        processed += len(paragraphs_rows)
        paragraphs_rows = []
    if page_rows:
        page_writer.writerows(page_rows)
        processed_pages += len(page_rows)
        page_rows = []

#write metadate into txt
avg_paragraph_length = tokens_sum / processed
avg_page_length = tokens_sum / processed_pages
chunking_meta_outputs = [
    f"Paragraphs processed in total: {processed}",
    f"Pages processed in total: {processed_pages}",
    f"empty paragraphs skipped: {empty_paragraphs}",
    f"Avg. page length: {avg_page_length}",
    f"Avg. paragraph length: {avg_paragraph_length}"
]
meta_file_path = "../data/chunking_metainfo.txt"
with open(meta_file_path, "w") as meta_output_txt:
    for i in chunking_meta_outputs:
        meta_output_txt.write(i+'\n')
        print(i)
print(f"Metainfo on chunking has been written to '{meta_file_path}'")





