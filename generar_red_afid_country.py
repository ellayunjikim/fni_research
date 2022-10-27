import sys
import os
import string

import pandas as pd

HELP_MESSAGE = 'Usage:\npython {} --data=<input_file> --output=<output_dir>'
INVALID_ARG_NUMBER = 'Expected 2 argument, but {} received.'
INSTRUCTIONS_MESSAGE = 'data: publications file'

AFID_NODES_FILE_NAME = "afid_nodes.csv"
AFID_EDGES_FILE_NAME = "afid_edges.csv"
COUNTRIES_NODES_FILE_NAME = "countries_nodes.csv"
COUNTRIES_EDGES_FILE_NAME = "countries_edges.csv"

def get_node_edges(row, dict_afid, edges_afid, dict_countries, edges_countries):
    try:
        afid_list = row['afid'].split(';')
        affilname_list = row['affilname'].split(';')
        affilcountry_list = row['affiliation_country'].split(';')
    except AttributeError:
        return
    for i in range(len(afid_list)):
        afid = afid_list[i]
        afid_name = affilname_list[i]
        country = affilcountry_list[i]
        if (afid not in dict_afid):
            if (country == ""):
                country = "Not found"
            dict_afid[afid] = [afid_name,country]
            dict_countries[country] = dict_countries.get(country, 0) + 1

    for i in range(len(afid_list)):
        for j in range(i+1, len(afid_list)):
            af_1 = afid_list[i]
            af_2 = afid_list[j]
            edges_afid[(af_1, af_2)] = edges_afid.get((af_1, af_2), 0) + 1
    
    for i in range(len(afid_list)):
        for j in range(i+1, len(afid_list)):
            ctry_1 = affilcountry_list[i]
            ctry_2 = affilcountry_list[j]
            if (ctry_1 == ""):
                ctry_1 = "Not found"
            if (ctry_2 == ""):
                ctry_2 = "Not found"
            if (ctry_1 != ctry_2):
                edges_countries[(ctry_1, ctry_2)] = edges_countries.get((ctry_1, ctry_2), 0) + 1
            
def generate_afid_files(dict_afid, edges_afid):
    afid_nodes_file = os.path.join(output_dir, f'{AFID_NODES_FILE_NAME}')
    afid_edges_file = os.path.join(output_dir, f'{AFID_EDGES_FILE_NAME}')

    nodes = open(afid_nodes_file, "w", encoding="utf-8")
    nodes.write("afid,name,country\n")
    for afid in dict_afid:
        nodes.write(str(afid)+","+dict_afid[afid][0]+","+dict_afid[afid][1]+"\n")
    nodes.close()
    
    edges = open(afid_edges_file, "w", encoding="utf-8")
    edges.write("source,target,weight\n")
    for key in edges_afid:
        string = str(key[0]) + "," + str(key[1]) + "," + str(edges_afid[key]) + "\n"
        edges.write(string)
    edges.close()
    
def generate_graph_files(data, output_dir):
    dict_afid = {}
    edges_afid = {}
    dict_countries = {}
    edges_countries = {}
    
    publications_df = pd.read_csv(input_file, sep='\t')
    publications_df.apply(lambda row: get_node_edges(row, dict_afid, edges_afid, dict_countries, edges_countries), axis=1)
    generate_afid_files(dict_afid, edges_afid)
    generate_countries_files(dict_countries, edges_countries)
    
def generate_countries_files(dict_countries, edges_countries):
    countries_nodes_file = os.path.join(output_dir, f'{COUNTRIES_NODES_FILE_NAME}')
    countries_edges_file = os.path.join(output_dir, f'{COUNTRIES_EDGES_FILE_NAME}')
    
    nodes = open(countries_nodes_file, "w", encoding="utf-8")
    nodes.write("country_id,name,num_affil\n")
    countries = list(dict_countries.keys())
    for idx, val in enumerate(countries):
        nodes.write(str(idx)+","+val+","+str(dict_countries[val])+"\n")
    nodes.close()

    edges = open(countries_edges_file, "w", encoding="utf-8")
    edges.write("source,target,weight\n")
    for key in edges_countries:
        string = str(countries.index(key[0])) + "," + str(countries.index(key[1])) + "," + str(edges_countries[key]) + "\n"
        edges.write(string)
    edges.close()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(INVALID_ARG_NUMBER.format(len(sys.argv) - 1))
        print(HELP_MESSAGE.format(sys.argv[0]))
        print(INSTRUCTIONS_MESSAGE)
        exit(1)
        
    input_file = ''
    output_dir = ''
    
    for arg in sys.argv[1:]:
        if arg.startswith('--data='): input_file = arg.split('=')[-1]
        elif arg.startswith('--output=') : output_dir = arg.split('=')[-1]

    generate_graph_files(input_file, output_dir)