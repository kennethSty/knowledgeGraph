#3rd party packages
import pandas as pd
# Intra-project packages
from utils.extract_utils import search_wiki

#define parameters for data extraction
search_params = {
    'list': 'search',
    'srprop': '',
    'srlimit': 5,  # Maximum number of results per query
    'limit': 10,  # Maximum number of results to return
    'srsearch': "Krankheit",
    'sroffset': 0  # Starting offset for pagination
}

#extract articles
pages = search_wiki(search_params=search_params, batch_proc=True)
pages_df = pd.DataFrame(pages)
pages_df.to_csv(f"../data/pages_until_sroff_{search_params['sroffset']}.csv", encoding="utf-8", index=False) #if locally: ../data