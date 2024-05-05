import ast
import csv
import sys
import re
import ast

import numpy as np


# needed for row_preprocessing
def increase_csv_maxsize():
    maxInt = sys.maxsize

    while True:
        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt / 10)

    csv.field_size_limit(sys.maxsize)

def extract_sections(text):
    # Define regular expression pattern to extract all titles
    title_pattern = r'(?<!=)==\s*([^=\n]+?)\s*==(?!=)'
    titles = re.findall(title_pattern, text)

    #Iterate over text by matching extracted titles and extract everything between as section
    sections = []
    counter = 0
    for i in range(len(titles) - 1):
        start_title = titles[i]
        end_title = titles[i + 1]
        counter+=1
        section_pattern = re.compile(
            r'==\s*' + re.escape(start_title) + r'\s*==\s*(.*?)==\s*' + re.escape(end_title) + r'\s*==', re.DOTALL)
        section_match = section_pattern.search(text)
        if section_match:
            sections.append(section_match.group(1).strip())

        # If the last title is reached, extract the section from that title to the end of the text
    if counter == len(titles) - 1:
        last_start_title = titles[-1]
        last_section_pattern = re.compile(r'==\s*' + re.escape(last_start_title) + r'\s*==\s*(.*)', re.DOTALL)
        last_section_match = last_section_pattern.search(text)
        if last_section_match:
            sections.append(last_section_match.group(1).strip())

    return sections, titles

def get_embedding_text(row, keys_to_embed_values):
    """

    :param
        row: A dictionary containing the content and metadata of an extracted Wikipedia Page
    :return:
        The preprocessed dictionary that includes "NA" if values are empty
    """

    # Make dict keys german fro German LLM
    ger_dict = {"section": "Inhalt", "section_title": "Untertitel", "page_title": "Titel", "summary": "Zusammenfassung", "title": "Titel"}

    embed_info_list = []
    # Read value of keys and modify to meaningful info strings
    for key in keys_to_embed_values:
        if type(row[key]) != str:
            unpacked_row_value = ast.literal_eval(row[key])
            if np.isnan(unpacked_row_value):
                row[key]="NA"
            elif type(unpacked_row_value)==list:
                row[key] = ", ".join(unpacked_row_value)
            else:
                row[key] = str(row[key])
        try:
            embed_info_list.append(ger_dict[key] + ": " + row[key])
        except ValueError:
            print("No german key specified in ger_dict to transform english key to")

    embedding_text = "\n".join(embed_info_list)

    return embedding_text

def extract_category(input_string):
    # Define a regex pattern to match "Kategorie:" followed by any characters
    pattern = r'Kategorie:(.+)'

    # Search for the pattern in the input string
    match = re.search(pattern, input_string)

    # If a match is found, extract the captured group
    if match:
        category = match.group(1).strip()  # Remove leading and trailing whitespaces
        return category
    else:
        return None  # Return None if "Kategorie:" is not found



