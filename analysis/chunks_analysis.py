import pandas as pd
import numpy as np

df = pd.read_csv("../data/02_preprocessed/chunked_pages.csv")
df_total = pd.read_csv("../data/02_preprocessed/total_pages2.csv")

print(df_total.describe())