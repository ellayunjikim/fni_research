import sys
import os
import string

import pandas as pd

HELP_MESSAGE = 'Usage:\npython {} --data=<input_file> --output=<output_dir>'
INVALID_ARG_NUMBER = 'Expected 2 argument, but {} received.'
INSTRUCTIONS_MESSAGE = 'data: publications file'

NODES_FILE_NAME = "nodes.csv"
EDGES_FILE_NAME = "edges.csv"

KW_FIELD = "authkeywords"
SEPARATOR = " | "

def get_node_edges(keywords, separator, list_nodes, dict_edges):
    try:
        list_keywords = keywords.lower().split(separator)
    except AttributeError:
        return
    for i in range(len(list_keywords)):
        keyword = list_keywords[i]
        keyword = keyword.translate(str.maketrans('', '', string.punctuation))
        if (keyword not in list_nodes):
            list_nodes.append(keyword)

    for i in range(len(list_keywords)):
        for j in range(i+1, len(list_keywords)):
            kw_1 = list_nodes.index(list_keywords[i].translate(str.maketrans('', '', string.punctuation)))
            kw_2 = list_nodes.index(list_keywords[j].translate(str.maketrans('', '', string.punctuation)))
            dict_edges[(kw_1, kw_2)] = dict_edges.get((kw_1, kw_2), 0) + 1
            
def generate_nodes_file(list_nodes, file):
    nodes = open(file, "w", encoding="utf-8")
    nodes.write("id,keyword\n")
    for idx, val in enumerate(list_nodes):
        nodes.write(str(idx)+","+val+"\n")
    nodes.close()
    
def generate_edges_file(dict_edges, file):
    edges = open(file, "w", encoding="utf-8")
    edges.write("source,target,weight\n")
    for key in dict_edges:
        string = str(key[0]) + "," + str(key[1]) + "," + str(dict_edges[key]) + "\n"
        edges.write(string)
    edges.close()
    
def generate_graph_files(data, output_dir):
    list_nodes = []
    dict_edges = {}
    nodes_file = os.path.join(output_dir, f'{NODES_FILE_NAME}')
    edges_file = os.path.join(output_dir, f'{EDGES_FILE_NAME}')

    publications_df = pd.read_csv(data, sep = '\t')
    publications_df[KW_FIELD].apply(lambda keywords: get_node_edges(keywords, SEPARATOR, list_nodes, dict_edges))
    generate_nodes_file(list_nodes, nodes_file)
    generate_edges_file(dict_edges, edges_file)

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
    