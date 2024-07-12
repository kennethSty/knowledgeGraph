import pandas as pd

########
df = pd.read_csv("../data/00_raw/pages_until_sroff_9750.csv")
no_summary = 0
total_summary = 0
yes_summary = 0
for row in df.summary:
    if pd.isnull(row):
        no_summary+=1
    else:
        total_summary+=len(row)
        yes_summary+=1

df_without_summary_na = df.dropna()
df_without_summary_na['summary_length'] = df_without_summary_na['summary'].apply(len)
df['content_length'] = df['content'].apply(len)

print("\nSummary Length Statistics:")
print(df_without_summary_na['summary_length'].describe())

print("\nContent Length Statistics:")
print(df['content_length'].describe())

# Calculate word count for 'summary' and 'content'
df_without_summary_na['summary_word_count'] = df_without_summary_na['summary'].apply(lambda x: len(x.split()))
df['content_word_count'] = df['content'].apply(lambda x: len(x.split()))

print("\nSummary Word Count Statistics:")
print(df_without_summary_na['summary_word_count'].describe())

print("\nContent Word Count Statistics:")
print(df['content_word_count'].describe())

print(f"Total Pages: {len(df.summary )}")
print(f"Pages with missing value: {df.isnull().sum()}")
print(f"No of pages w.o summary: {no_summary}")
print(f"Avg text length of summary: {total_summary/yes_summary}")
