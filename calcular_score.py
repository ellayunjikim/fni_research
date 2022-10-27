from datetime import date
import sys
import os

import pandas as pd
import numpy as np
from gensim import models
from pybliometrics.scopus import ScopusSearch
from preprocess import preprocess
from tqdm import tqdm

HELP_MESSAGE = 'Usage:\npython {} --data=<data_folder> --vocab=<vocabulary_file> --model=<w2v_file>'
INVALID_ARG_NUMBER = 'Expected 2 argument, but {} received.'
INSTRUCTIONS_MESSAGE = 'data: path to output files of calcular_produccion_academica.py\nvocab: research area words\nmodel: word embeddings file'

YEAR_TO = date.today().year - 1
YEAR_FROM = YEAR_TO - 5
QUERY = "AU-ID({}) AND PUBYEAR AFT {} AND PUBYEAR BEF {}"

#Files
RESEARCHERS_FILE = "researchers.csv"
ALL_PUBLICATIONS_FILE = 'all_publications.csv'

def load_resources(model_file, vocab_file):
    model = models.Word2Vec.load(model_file)
    vocab = pd.read_csv(vocab_file)
    vocab = list(vocab[:30]["term"])
    return model, vocab
    
def get_papers(scopus_id):
    request = True
    query = QUERY.format(scopus_id,YEAR_FROM-1,YEAR_TO+1)
    while(request):
        try:
            publications = ScopusSearch(query)
            request = False
        except ConnectionError:
            print("Connection error, retrying in 10 seconds")
            sleep(10)
    return pd.DataFrame(pd.DataFrame(publications.results))

def get_all_papers(researchers_list):
    pubs_list = []
    for i in tqdm(range(len(researchers_list)), desc = 'Paper search progress'):
        publications = get_papers(researchers_list[i])
        pubs_list.append(publications)
    pubs_df = pd.concat(pubs_list)
    pubs_df.drop_duplicates(subset='eid', inplace=True)
    pubs_df.dropna(subset=["eid"], inplace=True)
    return pubs_df
    
def publication_score(tokens, w2v_model, vocabulary):
    token_score = []
    for token in tokens:
        similarity = 0
        n_similarity = -1
        for crys_word in vocabulary:
            if (token in w2v_model.wv and crys_word in w2v_model.wv):
                n_similarity = w2v_model.wv.similarity(token, crys_word)
            if (n_similarity > 0 and n_similarity > similarity):
                similarity = n_similarity
        token_score.append(similarity)
    return sum(token_score)/len(token_score)
    
def researcher_score(researcher_id, publications):
    researcher_pubs = publications.loc[publications.author_ids.str.contains(str(researcher_id), na=False)]
    if (len(researcher_pubs)==0):
        return 0
    researcher_pubs = researcher_pubs[~researcher_pubs["description"].isna()]
    if (len(researcher_pubs)==0):
        return 0
    
    score = researcher_pubs["score"].mean()
    return score
    
def get_score(data_folder, model_file, vocabulary_file): 
    researchers_df = pd.read_csv(os.path.join(data_folder, f'{RESEARCHERS_FILE}'))
    w2v_model,vocab = load_resources(model_file, vocabulary_file)
    
    pubs_df = get_all_papers(researchers_df["scopusID"])
    print("Papers preprocessing step")
    pubs_df["tokens"] = pubs_df["description"].progress_apply(lambda abstract: preprocess(abstract) if (pd.notnull(abstract)) else None)
    pubs_df["score"] = pubs_df["tokens"].apply(lambda tokens: publication_score(tokens, w2v_model, vocab) if (np.all(pd.notnull(tokens))) else 0)
    
    researchers_df['score'] = researchers_df['scopusID'].apply(lambda researcher_id: researcher_score(researcher_id, pubs_df))
    researchers_df.to_csv(os.path.join(data_folder, f'{RESEARCHERS_FILE}'), index=False)
    pubs_df.to_csv(os.path.join(data_folder, f'{ALL_PUBLICATIONS_FILE}'), index=False, sep='\t')

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(INVALID_ARG_NUMBER.format(len(sys.argv) - 1))
        print(HELP_MESSAGE.format(sys.argv[0]))
        print(INSTRUCTIONS_MESSAGE)
        exit(1)
        
    data_folder = ''
    output_dir = ''
    vocab = ''
    model = ''
    
    for arg in sys.argv[1:]:
        if arg.startswith('--data='): data_folder = arg.split('=')[-1]
        elif arg.startswith('--vocab=') : vocab = arg.split('=')[-1]
        elif arg.startswith('--model=') : model = arg.split('=')[-1]
        
    tqdm.pandas()
    get_score(data_folder, model, vocab)
