import pandas as pd
import ast

with open("../data/04_eval/mesh_de_total.txt", 'r') as eval_file:
    total_mesh_string = eval_file.read()
    total_mesh = ast.literal_eval(total_mesh_string)
    total_mesh_set = set(total_mesh)

df = pd.read_csv("../data/00_raw/pages_until_sroff_9750.csv")

eval_df = pd.DataFrame()
while(len(eval_df)<10):
    row = df.sample()
    title = row.iloc[0]['title'].lower()
    if(title in total_mesh_set):
        eval_df = pd.concat([eval_df, row], ignore_index=True)

eval_df.to_csv(f"../data/00_raw/eval_pages_raw.csv", encoding="utf-8", index=False) #if locally: ../data

