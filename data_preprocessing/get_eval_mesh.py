import json

# Note not in data types: mesh:Descriptor, mesh:Concept, mesh:Qualifier
# Note not translated: All mesh:SCR_* -> possible translation necessary

#Open files for reading and writing
read_path = './data/MeSH_DE.jsonld'
write_path_json = './data/mesh_de_categories.json'
write_path_txt = './data/mesh_de_total.txt'

with open(read_path, 'r') as read_file, \
        open(write_path_json, 'w') as write_file_json, \
            open(write_path_txt, 'w') as write_file_txt:
    mesh = json.load(read_file)


    #Create dictionary of different data categories
    type_set = set()
    for entry in mesh.get('@graph', []):
        # Catches Terms of type: mesh:AllowedDescriptorQualifierPair, mesh:SCR
        type_set.add(entry["@type"])
    data_type_dict = {key: set() for key in type_set}

    #Normal list for capturing the data unstructured
    de_mesh = set()


    #Iterate over MeSH and insert german entities into categories of data_type_dict + de_mesh
    for entry in mesh.get('@graph', []):

        # Catches german terms of all categories except mesh:Term
        # if label is list iterate over dicts in list and extract the german version of the term
        if('label' in entry):
            label_dict = entry['label']
            if(isinstance(label_dict, list)):
                for term_dict in label_dict:
                    if term_dict.get('@language')=='de':
                        for key in data_type_dict.keys():
                            if key in entry['@type']:
                                data_type_dict[key].add(term_dict.get('@value').lower().strip())
                        de_mesh.add(term_dict.get('@value').lower().strip())

            elif(label_dict.get('@language')=='de'):
                        for key in data_type_dict.keys():
                            if key in entry['@type']:
                                data_type_dict[key].add(term_dict.get('@value').lower().strip())
                        de_mesh.add(term_dict.get('@value').lower().strip())


        ## Catches german mesh:Terms
        elif('prefLabel' in entry):
            pref_label_dict = entry['prefLabel']
            if(isinstance(pref_label_dict, list)):
                for term_dict in pref_label_dict:
                    if(term_dict.get('@languange'=='de')):
                        for key in data_type_dict.keys():
                            if key in entry['@type']:
                                data_type_dict[key].add(term_dict.get('@value').lower().strip())
                        de_mesh.add(term_dict.get('@value').lower().strip())


            elif(pref_label_dict.get('@language')=='de'):
                for key in data_type_dict.keys():
                    if key in entry['@type']:
                        data_type_dict[key].add(pref_label_dict.get('@value').lower().strip())
                de_mesh.add(pref_label_dict.get('@value').lower().strip())


    #writing data to files (convert cateogie sets to list for json serializability)
    data_type_dict_json = {key: list(value) for key, value in data_type_dict.items()}
    json.dump(data_type_dict_json, write_file_json, indent = 4)
    write_file_txt.write(str(de_mesh))


print(f"Total unique german terms (by Categories): {[{key, len(data_type_dict[key])} for key in data_type_dict.keys()]}")
print(f"Total unique german terms (in List: {len(de_mesh)}")
print(f"Total original entries in mesh: {len(mesh['@graph'])}")