import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from nltk.corpus import stopwords
import spacy
from nltk.util import ngrams
from collections import Counter

nlp = spacy.load("de_core_news_sm")
german_stop_words = stopwords.words('german')

input_csv = open("../data/04_eval/selected_eval_embedded_chunks.csv")
reader = csv.DictReader(input_csv)

na = 0
tokens_per_section = []
i = 0
for row in reader:
    section = row["section"]
    doc = nlp(section)
    section_tokens = [token.text for token in doc]
    tokens_per_section.append(section_tokens)


flattened_tokens = [token for token_list in tokens_per_section for token in token_list]
filtered_tokens = [word.lower() for word in flattened_tokens if word.isalpha() and word.lower() not in german_stop_words]

# Specify the desired n-gram size
n = 2

# Compute n-grams
n_grams = list(ngrams(filtered_tokens, n))

# Count the occurrences of each n-gram
n_gram_counts = Counter(n_grams)

# Display the most common n-grams
print(f"Most common {n}-grams:")
i = 0
for n_gram, count in n_gram_counts.most_common():
    print(f"{n_gram}: {count}")
    i+=1
    if count < 2:
        break

#Plot most common ngrams
# Create a DataFrame for visualization
top_ngrams = n_gram_counts.most_common(10)
df_top_ngrams = pd.DataFrame(top_ngrams, columns=['N-gram', 'Count'])

colors = sns.color_palette("viridis", 4)
plt.figure(figsize=(10, 6))
plt.barh([str(ngram) for ngram in df_top_ngrams['N-gram']], df_top_ngrams['Count'], color=colors)
plt.title(f'Most Common {n}-grams', color='white')
plt.xlabel('Count', fontsize=14, color='white')
plt.xticks(ha='right', color='white')  # Rotate x-axis labels for better readability
plt.yticks(color='white')
plt.grid(alpha=0.6, linestyle='--')
plt.tight_layout()

# Save the plot
plt.savefig('eval_data_1grams.png', dpi=300, transparent=True)

