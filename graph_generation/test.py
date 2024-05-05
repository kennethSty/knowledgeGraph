import csv
embedded = []
processed_rows = 0
with open("../data/small_embedded_chunks2.csv") as input_csv:
    reader = csv.DictReader(input_csv)
    for row in reader:
        processed_rows +=1
        embedded.append(row)
print(f"processed rows: {processed_rows}")
print(f"embedded in document:{len(embedded)}")

