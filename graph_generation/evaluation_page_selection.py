import pandas as pd

#read
chunks_df = pd.read_csv("/Users/Kenneth/PycharmProjects/knowledgeGraph/data/03_model_input/eval_embedded_chunks.csv")
pages_df = pd.read_csv("/Users/Kenneth/PycharmProjects/knowledgeGraph/data/03_model_input/eval_embedded_pages.csv")

#filter
chunks_df_filtered = chunks_df.loc[chunks_df['page_title'].isin(['Milzbrand', 'Lissenzephalie', 'Lindan', 'Ryanodin-Rezeptoren', 'Dermatitis'])]
pages_df_filtered = pages_df.loc[pages_df['title'].isin(['Milzbrand', 'Lissenzephalie', 'Lindan', 'Ryanodin-Rezeptoren', 'Dermatitis'])]

#write
chunks_df_filtered.to_csv("/Users/Kenneth/PycharmProjects/knowledgeGraph/data/04_eval/selected_eval_embedded_chunks.csv",
                          index = False)
pages_df_filtered.to_csv("/Users/Kenneth/PycharmProjects/knowledgeGraph/data/04_eval/selected_eval_embedded_pages.csv",
                         index = False)
print(f"Number of Sections: {len(chunks_df_filtered)}")
print(f"Titles filtered pages df: {pages_df_filtered['title'].unique()}")
print(f"Titles filtered section df: {chunks_df_filtered['page_title'].unique()}")

