
# Intra-project packages
from utils.extract_utils import set_user_agent, set_lang, search_wiki

#define parameters for data extraction
search_params = {
    'list': 'search',
    'srprop': '',
    'srlimit': 10,  # Maximum number of results per query
    'limit': 10,  # Maximum number of results to return
    'srsearch': "Krankheit",
    'sroffset': 0  # Starting offset for pagination
}

#extract articles
pages = search_wiki(search_params=search_params, batch_proc=True)