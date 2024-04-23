# Third party packages
import requests
import time
from datetime import datetime, timedelta
import wikipedia as wp


## Global variables
API_URL = 'http://en.wikipedia.org/w/api.php'
USER_AGENT = 'wikipedia (https://github.com/goldsmith/Wikipedia/)'
RATE_LIMIT = True
RATE_LIMIT_MIN_WAIT = timedelta(milliseconds=50)
RATE_LIMIT_LAST_CALL = None

## Start of utility functions
def set_lang(prefix):
  '''
  Changing the language from english (default) is done by adding a prefix at the right spot in the URL
  Set prefix to one of the two-lettered prefixes in `list of all Wikipedias <http://meta.wikimedia.org/wiki/List_of_Wikipedias>`

  '''

  global API_URL #change local scoope to global -> URL will be globally adapted.
  API_URL = 'http://' + prefix.lower() + '.wikipedia.org/w/api.php'

def set_user_agent(user_agent_string):
  '''
  Set the User-Agent string to be used for all requests.

  Arguments:

  * user_agent_string - (string) a string specifying the User-Agent header
  '''
  global USER_AGENT
  USER_AGENT = user_agent_string

# helper function wrapping the wiki API request
def wiki_request(params):
  '''
  Make a request to the Wikipedia API using the given search parameters.
  Returns a parsed dict of the JSON response.
  '''
  global RATE_LIMIT_LAST_CALL
  global USER_AGENT

  params['format'] = 'json'
  if not 'action' in params:
    params['action'] = 'query'

  headers = {
    'User-Agent': USER_AGENT
  }

  if RATE_LIMIT and RATE_LIMIT_LAST_CALL and \
    RATE_LIMIT_LAST_CALL + RATE_LIMIT_MIN_WAIT > datetime.now():

    # it hasn't been long enough since the last API call
    # so wait until we're in the clear to make the request

    wait_time = (RATE_LIMIT_LAST_CALL + RATE_LIMIT_MIN_WAIT) - datetime.now()
    time.sleep(int(wait_time.total_seconds()))

  r = requests.get(API_URL, params=params, headers=headers)
  print(r.url)

  if RATE_LIMIT:
    RATE_LIMIT_LAST_CALL = datetime.now()

  return r.json()

def search_wiki(search_params, batch_proc = True):
  """

  :param search_params:
    parameters specifying the query for which articles are retrieved
  :param batch_proc:
    controls whether srof parameter is used such that we keep extracting data after the limit is reached in batches
  :return:
    list of dictionaries containing metadata and content of the retrieved pages (1 dict per page)

  """
  set_lang('de')
  pages_list = []
  while len(pages_list) < search_params["limit"]:
      raw_results = wiki_request(search_params)
      total_pages = raw_results["query"]["searchinfo"]["totalhits"]

      # Add pages from the last query
      for p in raw_results["query"]["search"]:
          page_id = p["pageid"]
          title = p["title"]
          value_exists = any(d["pageid"] == page_id for d in pages_list)

          #avoid duplicate API queries for performance
          if value_exists==False:
            try:
              wp.set_lang('de') #set language of wp package to de
              page = wp.page(pageid=page_id)
              content = page.content
              page_dict = {"title": title, "pageid": page_id, "content": content}
              pages_list.append(page_dict)
              print(f"{len(pages_list)}")

            except wp.exceptions.DisambiguationError as e:
              print(f"Skipping disambiguation page: {title}")
          else:
            print("Duplicate id skipped: {page_id}")

      #if sroffset continue extracting pages in batches
      if batch_proc:
        # Update sroffset for the next iteration to crawl next batch
        search_params['sroffset'] = raw_results["continue"]["sroffset"]
        print(f"continue:{search_params['sroffset']}")

  print(f"Number of articles retrieved: {len(pages_list)}")
  return pages_list