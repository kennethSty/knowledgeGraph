#Third party libraries
import torch
import csv

#user libraries
from utils import embed_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = embed_utils.GerMedBert(device)

#TODO: Here we need to read the chunked documents instead!
with open("../data/oldpages_until_sroff_9750.csv") as input_csv, \
    open("../data/embedded_pages_until_sroff_9750.csv") as output_csv:
    reader = csv.DictReader(input_csv)
    writer = csv.DictWriter(output_csv, fieldnames=["page_id", "title", "content", "links","categories","head_sections","summary"])

for row in reader:
    page_content = row["content"]
    cls_token = model.embed(page_content)
    break



