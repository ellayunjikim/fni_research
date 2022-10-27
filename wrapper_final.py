#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Author: Zihao Yu 
# Email: yuzihao@grinnell.edu
# Last edit date: 2022/Aug/5th



# In[4]:


def connect_scopus_api(api_keys, insttoken = None):
    import pybliometrics#The first time calling import pybliometrics, it will have an interface asking the user to enter relevent information
    
    pybliometrics.scopus.utils.create_config([api_keys], insttoken)


# In[5]:


#Import the py files in the same directory
from calcular_produccion_academica import get_collaboration
from obtener_publicaciones import download_publications 
from generar_red_keywords import generate_graph_files
from preprocess_corpus import preprocess
from word_embeddings import create_model
from calcular_score import get_score

#Import packages
from time import sleep
from datetime import datetime
from datetime import date
import sys
import os
import pandas as pd
from pybliometrics.scopus import ScopusSearch
from pybliometrics.scopus import AuthorRetrieval
import glob
import re
import string
import nltk
from nltk import corpus
from nltk import FreqDist
from nltk.tokenize import word_tokenize
import numpy as np
import spacy
from gensim import models
from preprocess import preprocess
from tqdm import tqdm
import json
import requests
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
from sklearn.preprocessing import StandardScaler
from gensim.utils import simple_preprocess
from gensim import models
from gensim.corpora import Dictionary
from sklearn.decomposition import PCA
from collections import Counter
from nltk.stem.porter import *
np.random.seed(2018)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger') 
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from itertools import combinations
import pandas as pd
import warnings; warnings.simplefilter('ignore')
import json
import numpy as np

#Import packages

from scholarly import scholarly, ProxyGenerator
import csv
import math
import re
from collections import Counter
import json
from collections import defaultdict
#The NLP package
from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer
import yake

import pytextrank


import bibtexparser


# In[6]:


#To generate the vocabulary file

# Change corpus_stopwords with stopwords pertinent to your corpus or words not desired after a previous run of this notebook
corpus_stopwords = ['also','many','find','well','say']

def sent_to_words(sentences):
    for sentence in sentences:
        yield (simple_preprocess(str(sentence),deacc=True))
        
stopwords = nltk.corpus.stopwords.words('english')
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if (len(word)>3) and (word not in stopwords) and (word not in corpus_stopwords) ] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]
def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def createMatrix(dic,doc,corpus):
    matriz= np.zeros((len(doc),len(dic)))
    for doc in range(len(doc)):
        for word in range(len(corpus[doc])):
            matriz[doc][(corpus[doc][word][0])]+= corpus[doc][word][1]  
    return pd.DataFrame(matriz)


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def dictionary(data, filter_extreme=True,below=5, above=0.5):
    id2word = Dictionary(data)# sin filtroooooo
    if filter_extreme:
        id2word.filter_extremes(no_below= below, no_above= above, keep_n=100000)
        return id2word
    return id2word
        
    
def created_corpus(data, id2word):
    corpus = [id2word.doc2bow(text) for text in data] # Bag of words
    return corpus



def top_of_words(data,id2word,corpus,save=True):
        bow=createMatrix(dic=id2word,doc= data_words_nostops , corpus=corpus)
        lista_palabras=[]
        for i in range(len(id2word)):
            lista_palabras.append(id2word[i])
        bow.columns = lista_palabras
        top_words= bow.sum(axis=0).sort_values(ascending=False) 
        words=pd.DataFrame(top_words.head(10)).reset_index()
        words.columns= ["words", "count"]
        words["country"]=pd.DataFrame([country]*10)
        words["year"]=pd.DataFrame([year]*10)
        listaTops.append(words)
        if save:
            words.to_csv(country+"_"+str(year)+".csv")
            
            
def format_topics_sentences(ldamodel, corpus, texts):
    sent_topics_df = pd.DataFrame()
    
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list            
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num)+1, round(prop_topic,4), topic_keywords, year]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords', "year"]
    sent_topics_df["country"]=pd.DataFrame([country]*len(noticias))
    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)                


def topic_modeling(papers,abstract_column ):  

        doc_words= list(sent_to_words(papers[abstract_column].values))
        data_words_nostops= remove_stopwords(doc_words)
        data_lemmatized= lemmatization(data_words_nostops, allowed_postags=['NOUN', 'ADJ', 'VERB', "ADV"]) 
        data_preprocess= data_lemmatized

        id2word= dictionary(data= data_preprocess, filter_extreme=True,below=5, above=0.5)
        corpus= created_corpus(data=data_preprocess, id2word=id2word)
        

        #model 
        tfidf= models.TfidfModel(corpus)
        corpus_tfidf= tfidf[corpus]
        #Para modificaciones del ldaModel
        lda_model_TFIDF = models.LdaMulticore(corpus=corpus_tfidf,id2word=id2word,num_topics=5,random_state=100,chunksize=50,passes=10,per_word_topics=True, workers=4)
        return lda_model_TFIDF


def preprocess(papers,abstract_column):  

        doc_words= list(sent_to_words(papers[abstract_column].values))
        data_words_nostops= remove_stopwords(doc_words)
        data_lemmatized= lemmatization(data_words_nostops, allowed_postags=['NOUN', 'ADJ', 'VERB', "ADV"]) 
        data_preprocess= data_lemmatized

        id2word= dictionary(data= data_preprocess, filter_extreme=True,below=5, above=0.5)
        corpus= created_corpus(data=data_preprocess, id2word=id2word)
        return [id2word, corpus,data_lemmatized ]
 

#This is the function I wrote based on the python notebook
def get_vocabulary(corpus_path, vocabulary_path):
    
    data_all= pd.read_csv(corpus_path, engine='python', sep='\t\t')
    data_all = data_all.rename(columns={"text": 'description'})
    id2word,corpus,data_lemantized= preprocess(papers=data_all,abstract_column='description')
    tfidf  = models.TfidfModel(corpus)
    corpus_tfidf= tfidf[corpus]
    
    sentences_lematized = [" ".join(token) for token in data_lemantized]
    corpus_lematized = ' '.join(sentences_lematized) 
    freq = Counter(corpus_lematized.split()).most_common()
    
    term_freq_df =pd.DataFrame(freq, columns=['term', 'freq'])
    term_freq_df.to_csv(vocabulary_path, index=False)


# In[7]:


#From metric_extraction.ipynb
def get_rAPI(URL):
    '''
        create a request for a specific URL and return the response in a json
        :param URL:
        :return: json
    '''
    r = requests.get(
        url=URL,
        headers={
            'Accept': 'application/json'
        }
    )
    if r.status_code == 200:
        logging.info("Search successful")
        return r.json()
    elif r.status_code == 429:
        logging.error("Rate limit reached")
    elif r.status_code == 401:
        logging.error("Unauthorized, check WiFi Network")
    elif r.status_code == 400:
        logging.error("Bad request")
    else:
        logging.error(r.status_code)
        logging.info(r.json())
    print(URL)
    return None


def get_researchers_metrics(researchers ,areaURI ,metric ,time_range,byYear="false"):
    '''
        create a scival request for researchers metrics in a specific area and time range
        :param URL:
        :return: json
    '''
    query_url = URL.format(metric, researchers, time_range, areaURI, byYear, APIKEY)
    return get_rAPI(query_url)

def get_researchers_metrics_df(researchers ,areaURI="",metrics="ScholarlyOutput,FieldWeightedCitationImpact,HIndices" ,time_range="5yrsAndCurrent"):
    '''
    call to the scival's author metric, approx 100 authors for each QS area 5 years time range
    :param autores:
    :param areaURI:
    :param metrics:
    :param time_range:
    :return: dataframe which columns are scopusID, metrica(CitatioCount||ScholarlyOutput,
    nombre, area and years that belong to the time range added as param, each number of each year correspond to
    the number of publication on citations repective
    '''
    metric_json = get_researchers_metrics(researchers, areaURI, metrics, time_range)
    resultL = metric_json["results"]
    author_df = pd.DataFrame()
    for iauthor, result in enumerate(resultL):
        researcher_id = result["author"]["id"]
        name = result["author"]["name"]

        for metricD in result["metrics"]:
            metric = metricD["metricType"]  # ScholarlyOutput | CitationCount
            if "value" in metricD.keys() :
                value = metricD["value"]
            else:
                value = np.nan
            tmp = pd.DataFrame({"metric":[metric],"scopusID":[researcher_id],"name":[name],"value":[value]})
            author_df = pd.concat([author_df, tmp])
    return author_df

def get_network_coauthors(researcher_id, papers_df, threshold=0.7):
    '''
        get the amount of co-authors for each researcher
        :param researcher_id: researcher's Scopus ID
        :param papers_df: dataframe which contains every publication in the last five years for every researcher
        :param threshold: minimum score for considering a paper pertinent to the network
        :return: int
    '''
    researcher_pubs = papers_df.loc[papers_df["author_ids"].str.contains(researcher_id)]
    researcher_pubs = researcher_pubs.loc[researcher_pubs["score"]>=threshold]
    if (len(researcher_pubs)==0):
        return 0
    list_coauthors = researcher_pubs["author_ids"].str.split(";").sum()
    return len(set(list_coauthors)) - 1

#This is the function I wrote base on their notebook
def metric_extraction(apikey_file, researchers_file, all_pubs_file):
    
    researchers_df = pd.read_csv(researchers_file, dtype={"scopusID":str})
    all_pubs_df = pd.read_csv(all_pubs_file, sep='\t')
    
    with open(apikey_file, "r") as f:
        APIKEY = f.readline()
    URL = ("https://api.elsevier.com/analytics/scival/author/metrics?metricTypes={}&"
            "authors={}&"
            "yearRange={}&"
            "subjectAreaFilterURI={}&"
            "includeSelfCitations=true&"
            "byYear={}&"
            "includedDocs=AllPublicationTypes&"
            "journalImpactType=CiteScore&"
            "showAsFieldWeighted=false&"
            "indexType=hIndex&"
            "apiKey={}")

    batch_size = 100
    ilote = 0
    isTerminarRecorrido = ilote *batch_size < researchers_df.shape[0]
    metrics = "ScholarlyOutput,FieldWeightedCitationImpact,HIndices"
    time_range = "5yrsAndCurrent"
    aufinaldf = pd.DataFrame()
    while isTerminarRecorrido:
        researchers = ",".join(researchers_df[ilote *batch_size:(ilote +1 ) *batch_size]["scopusID"])
        researcher_df = get_researchers_metrics_df(researchers ,"" ,metrics ,time_range)
        aufinaldf = pd.concat([aufinaldf, researcher_df])
        ilote += 1
        isTerminarRecorrido = ilote *batch_size < researchers_df.shape[0]
        
    metrics_df = aufinaldf.pivot_table(index=['scopusID', 'name'],columns=['metric'],values="value").reset_index()
    metrics_df["scopusID"] = metrics_df["scopusID"].astype(str)
    final_df= pd.merge(researchers_df, metrics_df[["scopusID","FieldWeightedCitationImpact","HIndices","ScholarlyOutput"]], on="scopusID")
    final_df["num_network_coauthors"] = final_df["scopusID"].apply(lambda author_id: get_network_coauthors(author_id, all_pubs_df))
    final_df['HIndices'] = final_df['HIndices'].astype(int)
    final_df['ScholarlyOutput'] = final_df['ScholarlyOutput'].astype(int)
    final_df.to_csv(authors_file, index=False)
    
#from cluster_analysis.ipynb
def escalar(column):
    scaler = MinMaxScaler()
    return scaler.fit_transform(column)

def cluster_analysis(authors_file):
    
    authors_df = pd.read_csv(authors_file)
    authors_df = authors_df.loc[authors_df["ScholarlyOutput"]!=0]
    authors_df = authors_df.loc[authors_df["score"]!=0]
    
    cluster_attributes = authors_df[["FieldWeightedCitationImpact","ScholarlyOutput","HIndices","score","num_network_coauthors"]]
    cluster_attributes["FieldWeightedCitationImpact"] = cluster_attributes["FieldWeightedCitationImpact"].fillna(0)
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(cluster_attributes.values)
    
    distortions = []
    K = range(1,10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k, random_state=123)
        kmeanModel.fit(scaled_features)
        distortions.append(kmeanModel.inertia_)
        
    plt.figure(figsize=(16,8))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()
    
    optimal_k = 5
    kmeans = KMeans(n_clusters=optimal_k, random_state=123)
    labels = kmeans.fit_predict(scaled_features)
    
    pca = PCA(2)
    authors_2d = pca.fit_transform(scaled_features)
    plt.figure(figsize=(20, 10), dpi=80)

    #filter rows of original data
    filtered_label0 = authors_2d[labels == 0]
    filtered_label1 = authors_2d[labels == 1]
    filtered_label2 = authors_2d[labels == 2]
    filtered_label3 = authors_2d[labels == 3]
    filtered_label4 = authors_2d[labels == 4]

    #Plotting the results
    plt.scatter(filtered_label0[:,0] , filtered_label0[:,1] , color = 'red')
    plt.scatter(filtered_label1[:,0] , filtered_label1[:,1] , color = 'yellow')
    plt.scatter(filtered_label2[:,0] , filtered_label2[:,1] , color = 'blue')
    plt.scatter(filtered_label3[:,0] , filtered_label3[:,1] , color = 'green')
    plt.scatter(filtered_label4[:,0] , filtered_label4[:,1] , color = 'black')
    plt.show()
    
    authors_df["label"] = labels
    
    authors_df['FWCI_scaled'] = escalar(authors_df['FieldWeightedCitationImpact'].values.reshape(-1, 1))
    authors_df['h-index_scaled'] = escalar(authors_df['HIndices'].values.reshape(-1, 1))
    authors_df['n_coauthors_scaled'] = escalar(authors_df['num_network_coauthors'].values.reshape(-1, 1))
    
    authors_df[['FWCI_scaled','h-index_scaled','score','n_coauthors_scaled']].plot.box()
    
    group = 3
    important_cols_1 = ['scopusID','nombre']
    important_cols_2 = ['FWCI_scaled','h-index_scaled','score','n_coauthors_scaled']

    authors_df.loc[authors_df["label"]==group][important_cols_1].head(15)
    
    authors_df.loc[authors_df["label"]==group][important_cols_2].plot.box(figsize=(7,10))
    
    final_df = authors_df.drop(['FWCI_scaled', 'h-index_scaled', 'n_coauthors_scaled'], axis=1)
    final_df.rename(columns = {'FieldWeightedCitationImpact':'FWCI', 'label':'group_id'}, inplace = True)

    final_df.to_csv(authors_file, index=False)


# In[8]:


def export_two_df_to_one_json(df_nodes, df_edges, output_dir, author_or_keywords):
    
    list_nodes = []
    for i in range(0, len(df_nodes)):
        current_dict = {}
        for j in range(0, len(df_nodes.keys())):
            current_dict.update({'%s'%(df_nodes.keys()[j]) : dtype_adapter(df_nodes['%s'%(df_nodes.keys()[j])][i])})
    
        list_nodes.append(current_dict)
    
    list_edges = []
    for i in range(0, len(df_edges)):
        current_dict = {}
        for j in range(0, len(df_edges.keys())):
            current_dict.update({'%s'%(df_edges.keys()[j]) : dtype_adapter(df_edges['%s'%(df_edges.keys()[j])][i])})
            
        list_edges.append(current_dict)
    
    export_dict = {'nodes': list_nodes,
                   'links': list_edges}
    
    time = datetime.now()
    
    with open("%s_nodes_edges_%d_%d_%d_%d_%d_%d.json"%(author_or_keywords, time.year, time.month, time.day, time.hour, time.minute, time.second), "w") as outfile:
        json.dump(export_dict, outfile, indent = 4)


# In[9]:


#JSON will report 'Object of type int64 is not JSON serializable', so we need to convert every int64 to int
def dtype_adapter(input_data):
    
    new_data = input_data
    
    if type(input_data) == np.int64:
        new_data = int(input_data)
    
    return new_data


# In[10]:


def pairs(input_list):
    return list(combinations(input_list, 2))


# In[11]:


def check_not_exist(source, target, df_source_target):
    
    if df_source_target.empty:
        return True
    
    for i in range(0, len(df_source_target)):
        if (df_source_target['source'][i] == source and df_source_target['target'][i] == target) or (df_source_target['source'][i] == target and df_source_target['target'][i] == source):
            df_source_target['weight'][i] += 1
            return False
        
    #The source and target pair is not existed
    return True


# In[12]:


def node_not_exist(author_or_keywords_id, df_nodes):
    
    if df_nodes.empty:
        return True
    
    #If the author or keywords is already in the nodes, don't record it 
    for i in range(0, len(df_nodes)):
        if author_or_keywords_id == df_nodes['id'][i]:
            df_nodes['weight'][i] += 1
            return False
    
    return True


# In[13]:


def create_author_or_keywords_id_array(input_array, df_nodes, author_or_keywords):
    
    id_array = []
    for i in range(0, len(input_array)):
        for j in range(0, len(df_nodes)):
            if input_array[i] == df_nodes['%s'%(author_or_keywords)][j]:
                id_array.append(df_nodes['id'][j])
                  
    return id_array


# In[14]:


def create_author_or_keywords_id_column(df_pub, df_nodes, author_or_keywords, author_or_keywords_col_name_in_pub):
    
    df_pub['%s_id'%(author_or_keywords)] = ""
    for i in range(0, len(df_pub)):
        if df_pub['%s'%(author_or_keywords_col_name_in_pub)][i] != "":
            df_pub['%s_id'%(author_or_keywords)][i] = create_author_or_keywords_id_array(df_pub['%s'%(author_or_keywords_col_name_in_pub)][i], df_nodes, author_or_keywords)
    


# In[15]:


def generate_nodes_with_ids(df_pub, ids_col_name, names_or_keywords, author_or_keywords_col, id_separator, author_separator):
    
    for i in range(0, len(df_pub)):
        if type(df_pub['%s'%(ids_col_name)][i]) != float:
            df_pub['%s'%(ids_col_name)][i] = df_pub['%s'%(ids_col_name)][i].split('%s'%(id_separator))
        elif type(df_pub['%s'%(ids_col_name)][i]) == float:
            df_pub['%s'%(ids_col_name)][i] = ""
            
    for i in range(0, len(df_pub)):
        if type(df_pub['%s'%(author_or_keywords_col)][i]) != float:
            df_pub['%s'%(author_or_keywords_col)][i] = df_pub['%s'%(author_or_keywords_col)][i].split('%s'%(author_separator))
        elif type(df_pub['%s'%(author_or_keywords_col)][i]) == float:
            df_pub['%s'%(author_or_keywords_col)][i] = ""
    
    df_nodes = pd.DataFrame(columns = ['id', '%s'%(names_or_keywords), 'weight'])

    for i in range(0, len(df_pub)):
        for j in range(0, len(df_pub['%s'%(ids_col_name)][i])):#it is 'author_ids' for scopus
            if node_not_exist(df_pub['%s'%(ids_col_name)][i][j], df_nodes):
                df_nodes = df_nodes.append({'id': df_pub['%s'%(ids_col_name)][i][j], '%s'%(names_or_keywords): df_pub['%s'%(author_or_keywords_col)][i][j], 'weight': 1}, ignore_index = True)
                print('the pub now is', i)
                print(df_nodes)
    
    return df_nodes


# In[16]:


#From a
def generate_nodes_without_ids(df, names_or_keywords_col_name, names_or_keywords, separator):
    
    #Step 1: Separate the keywords in a str to many elements in an array
    keywords_long_list = []
    for i in range(0, len(df)):  
        if type(df['%s'%(names_or_keywords_col_name)][i]) != float and type(df['%s'%(names_or_keywords_col_name)][i]) != list and type(df['%s'%(names_or_keywords_col_name)][i]) != np.ndarray:
            df['%s'%(names_or_keywords_col_name)][i] = df['%s'%(names_or_keywords_col_name)][i].split('%s'%(separator))
            for j in range(0, len(df['%s'%(names_or_keywords_col_name)][i])):
                keywords_long_list.append(df['%s'%(names_or_keywords_col_name)][i][j])
        elif type(df['%s'%(names_or_keywords_col_name)][i]) == float:
            df['%s'%(names_or_keywords_col_name)][i] = ""
        elif type(df['%s'%(names_or_keywords_col_name)][i]) == list or type(df['%s'%(names_or_keywords_col_name)][i]) == np.ndarray:
            for x in range(0, len(df['%s'%(names_or_keywords_col_name)][i])):
                keywords_long_list.append(df['%s'%(names_or_keywords_col_name)][i][x])
            
            
    #Step 2: Assign id for each keyword and generate nodes dataframe
    d = defaultdict(lambda: len(d))
    omg_id = [d[x] for x in keywords_long_list]
    keywords_id = {'%s'%(names_or_keywords): keywords_long_list,
        'id': omg_id}
    
    
    df_keywords_id_raw = pd.DataFrame(keywords_id)

    df_keywords_id = pd.DataFrame(columns = ['id', '%s'%(names_or_keywords), 'weight'])

    for i in range(0, len(df_keywords_id_raw)):
        if node_not_exist(df_keywords_id_raw['id'][i], df_keywords_id):

            df_keywords_id = df_keywords_id.append({'id': df_keywords_id_raw['id'][i], '%s'%(names_or_keywords): df_keywords_id_raw['%s'%(names_or_keywords)][i], 'weight': 1}, ignore_index = True)
        print(df_keywords_id)
        
    return df_keywords_id


# In[17]:


def generate_edges(df_pub, ids):
    
    df_source_target = pd.DataFrame(columns = ['source', 'target', 'weight'])

    for i in range(0, len(df_pub)):
        current_pairs = pairs(df_pub['%s'%(ids)][i])
        #If it is not empty
        if len(current_pairs) != 0:
            for j in range(0, len(current_pairs)):
                if check_not_exist(current_pairs[j][0], current_pairs[j][1], df_source_target):
                    df_source_target = df_source_target.append({'source': current_pairs[j][0], 'target': current_pairs[j][1], 'weight': 1}, ignore_index=True)
                    print('the publication now is', i)
                    print(df_source_target)
    return df_source_target


# In[18]:


#JSON will report 'Object of type int64 is not JSON serializable', so we need to convert every int64 to int
def dtype_adapter(input_data):
    
    new_data = input_data
    
    if type(input_data) == np.int64:
        new_data = int(input_data)
    
    return new_data


# In[19]:


def export_two_df_to_one_json(df_nodes, df_edges, output_dir, author_or_keywords):
    
    list_nodes = []
    for i in range(0, len(df_nodes)):
        current_dict = {}
        for j in range(0, len(df_nodes.keys())):
            current_dict.update({'%s'%(df_nodes.keys()[j]) : dtype_adapter(df_nodes['%s'%(df_nodes.keys()[j])][i])})
    
        list_nodes.append(current_dict)
    
    list_edges = []
    for i in range(0, len(df_edges)):
        current_dict = {}
        for j in range(0, len(df_edges.keys())):
            current_dict.update({'%s'%(df_edges.keys()[j]) : dtype_adapter(df_edges['%s'%(df_edges.keys()[j])][i])})
            
        list_edges.append(current_dict)
    
    export_dict = {'nodes': list_nodes,
                   'links': list_edges}
    
    time = datetime.now()
    
    with open("%s\%s_nodes_edges_%d_%d_%d_%d_%d_%d.json"%(output_dir, author_or_keywords, time.year, time.month, time.day, time.hour, time.minute, time.second), "w") as outfile:
        json.dump(export_dict, outfile, indent = 4)


# In[20]:


#Sample usage from scopus: generate_author_files_with_defined_id(df_pub, 'author_ids', 'author_names', r'C:', ';', ';')

def generate_author_files_with_defined_id(df_pub, ids_col_name, author_col_name_in_pub, output_dir, id_separator, author_separator):
    
    #Generate nodes    
    df_author_nodes = generate_nodes_with_ids(df_pub, ids_col_name, 'authors', author_col_name_in_pub, id_separator, author_separator)
    
    #Generate edges
    create_author_or_keywords_id_column(df_pub, df_author_nodes, 'authors', author_col_name_in_pub)
    df_author_edges = generate_edges(df_pub, 'authors_id')
    
    #Export to JSON
    export_two_df_to_one_json(df_author_nodes, df_author_edges, output_dir, 'authors')
    print('The author files are created successfully')
    


# In[21]:


#Sample usage from scopus: generate_author_files_with_defined_id_to_csv(df_pub, 'author_ids', 'author_names', r'C:', ';', ';')

def generate_author_files_with_defined_id_to_csv(df_pub, ids_col_name, author_col_name_in_pub, output_dir, id_separator, author_separator):
    
    #Generate nodes    
    df_author_nodes = generate_nodes_with_ids(df_pub, ids_col_name, 'authors', author_col_name_in_pub, id_separator, author_separator)
    
    #Generate edges
    create_author_or_keywords_id_column(df_pub, df_author_nodes, 'authors', author_col_name_in_pub)
    df_author_edges = generate_edges(df_pub, 'authors_id')
    
    #Export to csv
    df_author_nodes.to_csv(output_dir + '\researchers.csv')
    df_author_edges.to_csv(output_dir + '\collabrations.csv')
    print('The author files are created successfully')


# In[22]:


#Sample usage for scopus: generate_author_files_without_defined_id_to_csv(df_pub, 'author_names', ';', r'C:')

def generate_author_files_without_defined_id_to_csv(df_pub, authors_col_name_in_pub, separator, output_dir):
    
    #Generate nodes
    df_author_nodes = generate_nodes_without_ids(df_pub, authors_col_name_in_pub, 'authors', separator)
    
    #Generate edges
    create_author_or_keywords_id_column(df_pub, df_author_nodes, 'authors', authors_col_name_in_pub)
    df_author_edges = generate_edges(df_pub, 'authors_id')
    
    #Export to csv
    df_author_nodes.to_csv(output_dir + '\researchers.csv')
    df_author_edges.to_csv(output_dir + '\collabrations.csv')
    print('The author files are created successfully')


# In[23]:


#Sample usage for scopus: generate_author_files_without_defined_id(df_pub, 'author_names', ';', r'C:')

def generate_author_files_without_defined_id(df_pub, authors_col_name_in_pub, separator, output_dir):
    
    #Generate nodes
    df_author_nodes = generate_nodes_without_ids(df_pub, authors_col_name_in_pub, 'authors', separator)
    
    #Generate edges
    create_author_or_keywords_id_column(df_pub, df_author_nodes, 'authors', authors_col_name_in_pub)
    df_author_edges = generate_edges(df_pub, 'authors_id')
    
    #Export to JSON
    export_two_df_to_one_json(df_author_nodes, df_author_edges, output_dir, 'authors')
    print('The author files are created successfully')


# In[24]:


#Sample usage for scopus: generate_author_files_without_defined_id(df_pub, 'author_names', ';', r'C:')

def generate_author_files_without_defined_id_to_csv(df_pub, authors_col_name_in_pub, separator, output_dir):
    
    #Generate nodes
    df_author_nodes = generate_nodes_without_ids(df_pub, authors_col_name_in_pub, 'authors', separator)
    
    #Generate edges
    create_author_or_keywords_id_column(df_pub, df_author_nodes, 'authors', authors_col_name_in_pub)
    df_author_edges = generate_edges(df_pub, 'authors_id')
    
    #Export to csv
    df_author_nodes.to_csv(output_dir + '\researchers.csv')
    df_author_edges.to_csv(output_dir + '\collabrations.csv')
    print('The author files are created successfully')


# In[25]:


#Sample usage for the dataset from Scopus: generate_keywords_files_with_defined_id(df_pub, 'keywords_id', 'authkeywords', r'C:')

def generate_keywords_files_with_defined_id(df_pub, ids_col_name, keywords_col_name_in_pub, output_dir):
    
    #Generate nodes
    df_keywords_nodes = generate_nodes_with_ids(df_pub, ids_col_name, 'keywords', keywords_col_name_in_pub)
    
    #Generate edges
    create_author_or_keywords_id_column(df_pub, df_keywords_nodes, 'keywords', keywords_col_name_in_pub)
    df_keywords_edges = generate_edges(df_pub, 'keywords_id')
    
    #Export to JSON
    export_two_df_to_one_json(df_keywords_nodes, df_keywords_edges, output_dir, 'keywords')
    print('The keywords files are created successfully')


# In[26]:


#Sample usage for the dataset from Scopus: generate_keywords_files_with_defined_id(df_pub, 'keywords_id', 'authkeywords', r'C:')

def generate_keywords_files_with_defined_id_to_csv(df_pub, ids_col_name, keywords_col_name_in_pub, output_dir):
    
    #Generate nodes
    df_keywords_nodes = generate_nodes_with_ids(df_pub, ids_col_name, 'keywords', keywords_col_name_in_pub)
    
    #Generate edges
    create_author_or_keywords_id_column(df_pub, df_keywords_nodes, 'keywords', keywords_col_name_in_pub)
    df_keywords_edges = generate_edges(df_pub, 'keywords_id')
    
    #Export to csv
    df_keywords_nodes.to_csv(output_dir + '\nodes.csv')
    df_keywords_edges.to_csv(output_dir + '\edges.csv')
    print('The keywords files are created successfully')


# In[27]:


#Sample usage for the dataset from Scopus: generate_keywords_files_without_defined_id(df_pub, 'authkeywords', ' | ', r'C:')

def generate_keywords_files_without_defined_id(df_pub, keywords_col_name_in_pub, separator, output_dir):
    
    #Generate nodes
    df_keywords_nodes = generate_nodes_without_ids(df_pub, keywords_col_name_in_pub, 'keywords', separator)
    
    #Generate edges
    
    create_author_or_keywords_id_column(df_pub, df_keywords_nodes, 'keywords', keywords_col_name_in_pub)
    df_keywords_edges = generate_edges(df_pub, 'keywords_id')
    
    #Export to JSON
    export_two_df_to_one_json(df_keywords_nodes, df_keywords_edges, output_dir, 'keywords')
    print('The keywords files are created successfully')


# In[28]:


#Sample usage for the dataset from Scopus: generate_keywords_files_without_defined_id(df_pub, 'authkeywords', ' | ', r'C:')

def generate_keywords_files_without_defined_id_to_csv(df_pub, keywords_col_name_in_pub, separator, output_dir):
    
    #Generate nodes
    df_keywords_nodes = generate_nodes_without_ids(df_pub, keywords_col_name_in_pub, 'keywords', separator)
    
    #Generate edges
    
    create_author_or_keywords_id_column(df_pub, df_keywords_nodes, 'keywords', keywords_col_name_in_pub)
    df_keywords_edges = generate_edges(df_pub, 'keywords_id')
    
    #Export to csv
    df_keywords_nodes.to_csv(output_dir + '\nodes.csv')
    df_keywords_edges.to_csv(output_dir + '\edges.csv')
    print('The keywords files are created successfully')


# In[29]:


def bibtex_to_df(absolute_path):
    
    with open(absolute_path, encoding = 'UTF-8') as bibtex_file:
        bib_database = bibtexparser.load(bibtex_file)

    dataframe = pd.DataFrame(bib_database.entries)
    
    return dataframe


# In[30]:


def connect_API_scraper(API_key):
    print('Connecting the ScraperAPI')
    pg = ProxyGenerator()
    success = pg.ScraperAPI(API_key)
    scholarly.use_proxy(pg)
    


# In[31]:


def search_pubs(keywords, year_range):
    
    #Step 1: query and convert the results type
    print("Start Step 1: query and convert the results type")
    search_query = scholarly.search_pubs(keywords)
    results_list = list(search_query)
    results_array = np.array(results_list)
    results_df_json_normalization = pd.json_normalize(results_array)
    
    
    return results_df_json_normalization   


# In[32]:


def pubs_time_range(results_df_json_normalization, year_range):
    
    #Step 2: Select the data after year_range. If year_range = 2012, then the data would be from 2013 to 2022.
    print("Start Step 2: Select the data after year_range.")
    df_ten_years = pd.DataFrame(columns = ['container_type', 'source', 'filled', 'gsrank', 'pub_url', 'author_id', 'url_scholarbib', 'url_add_sclib', 'num_citations', 'citedby_url', 'url_related_articles', 'bib.tltle', 'bib.author', 'bib.pub_year', 'bib.venue', 'bib.abstract', 'eprint_url'])

    for i in range(0, len(results_df_json_normalization)):
        if results_df_json_normalization['bib.pub_year'][i] != 'NA'and int(results_df_json_normalization['bib.pub_year'][i]) >= year_range[0] and int(results_df_json_normalization['bib.pub_year'][i]) <= year_range[1]: 
            df_ten_years = df_ten_years.append(results_df_json_normalization.iloc[i])
    
          
    return df_ten_years


# In[33]:


#title_col_name is bib.title and abstract_col_name is bib.abstract for google scholar
def generate_keywords(df_ten_years, title_col_name, abstract_col_name):
    
    #Step 3: Get keywords from title and abstract
    print("Start Step 3: Get keywords from title and abstract")
    ##Add one more colunm called keywords
    df_ten_years['keywords'] = "To be filled"
    
    #Start the NLP modeL
    print('Connect to the NLP model...')
    kw_model = KeyBERT(model='all-mpnet-base-v2')
    #Declare the vectorizer
    vectorizer = KeyphraseCountVectorizer()
    
    keywords_array = np.array([])
    keywords_long_list_keybert = np.array([])
    
    print('Start generating the keywords')
    for i in range(0, len(df_ten_years)):
        if type(df_ten_years['%s'%(title_col_name)][i]) != float:
            ##Check if the publication has abstract or not
            if type(df_ten_years['%s'%(abstract_col_name)][i]) != float:
                doc = df_ten_years['%s'%(title_col_name)][i] + '. ' + df_ten_years['%s'%(abstract_col_name)][i] ##I add a period and space between the title and abstract.
            else:
                doc = df_ten_years['%s'%(title_col_name)][i]

            results_two_words = kw_model.extract_keywords(doc, vectorizer=vectorizer)
            
            keywords_array = []

            for k in range(0, len(results_two_words)): 
                keywords_array.append(results_two_words[k][0])

            keywords_long_list_keybert = np.append(keywords_long_list_keybert, keywords_array)

        df_ten_years['keywords'][i] = keywords_array
            

        
    print('Finish generating keywords')
    
    return df_ten_years, keywords_long_list_keybert


# In[34]:


#Example usage: google_scholar_search('google_scholar', 'Perception of physical stability and center of mass of 3D objects', '75907e416291e41bef91fddcebe9973b', [2000, 2022], r'C:')
def google_scholar_search(search_engine, input_keywords, API_key, year_range, output_dir):
    
    
    if search_engine == 'google_scholar':
        #Step 1
        connect_API_scraper(API_key)#API_key is the scraperAPI
        #Step 2
        results_df_json_normalization = search_pubs(input_keywords, year_range)              
        df_pub = generate_keywords(pubs_time_range(results_df_json_normalization, year_range), 'bib.title', 'bib.abstract')[0]
        #Step 3 Export Data
            
        generate_author_files_without_defined_id(df_pub, 'bib.author', ',', output_dir)
        generate_keywords_files_without_defined_id(df_pub, 'keywords', ',', output_dir)

        print('------End------')
            
    elif search_engine == 'worldcat':
        print('The worldcat is not implemented')
    else:
        print('Sorry, the program only supports google_scholar and worldcat.')


# In[40]:


#Sample usage: generate_files_without_crys_score('csv', 'file_path', scopus, r'C:')
def generate_files_without_crys_score(bibtex_or_csv, pub_file_path, database, output_dir, abstract_col_name = None, title_col_name = None, keyword_provided = True, user_choice = False, author_separator = None, keywords_separator = None, author_id_separator = None, author_id = True, keyword_id = False, author_col_name = None, keywords_col_name = None, author_ids_col_name = None, keywords_ids_col_name = None):
    
    if bibtex_or_csv == 'bibtex':
        df_pub = bibtex_to_df(pub_file_path)
    elif bibtex_or_csv == 'csv':
        df_pub = pd.read_csv(pub_file_path)
    else:
        print('Only support bibtex and csv')
    
    if database == 'scopus':
        print('Start the process for scopus')
        
        generate_author_files_with_defined_id(df_pub, 'author_ids', 'author_names', output_dir, ';', ';')
        generate_keywords_files_without_defined_id(df_pub, 'authkeywords', ' | ', output_dir)
        
    elif database == 'acm':
        print('Start the process for acm')
        generate_author_files_without_defined_id(df_pub, 'author', ', ', output_dir)
        generate_keywords_files_without_defined_id(df_pub, 'keywords', ', ', output_dir)
    
    #This function is the complex version for user choice
    elif user_choice == True and author_separator != None and keywords_separator != None and author_id_separator != None:
        print('Start the user choice')
        
        if keyword_provided == False:
            #Generate keywords from title and abstract
            print('The keywords is not provided and start to generate the keywords')
            generate_keywords(df_pub, 'title', 'abstract')
            
        if author_id == True and keyword_id == True:
            generate_author_files_with_defined_id(df_pub, author_ids_col_name, author_col_name, output_dir, author_id_separator, author_separator)
            generate_keywords_files_with_defined_id_to_csv(df_pub, keywords_id_col_name, keywords_col_name, output_dir)
            
        elif author_id == False and keyword_id == False:
            generate_author_files_without_defined_id(df_pub, author_ids_col_name, author_col_name, output_dir, author_id_separator, author_separator)
            generate_keywords_files_without_defined_id(df_pub, keywords_col_name, keywords_separator, output_dir)
            
        elif author_id == True and keyword_id == False:
            generate_author_files_with_defined_id(df_pub, author_ids_col_name, author_col_name, output_dir, author_id_separator, author_separator)
            generate_keywords_files_without_defined_id(df_pub, keywords_col_name, keywords_separator, output_dir)
            
        elif author_id == False and keyword_id == True:
            generate_author_files_without_defined_id(df_pub, author_ids_col_name, author_col_name, output_dir, author_id_separator, author_separator)
            generate_keywords_files_with_defined_id_to_csv(df_pub, keywords_id_col_name, keywords_col_name, output_dir)
            
    else:
        print('the database is not supported, but you can access the root functions.')


# In[36]:


def generate_files_with_crys_score(bibtex_or_csv, pub_file_path, database, output_dir, apikey_file, corpus_file_path, abstract_col_name = None, title_col_name = None, keyword_provided = True, user_choice = False, author_separator = None, keywords_separator = None, author_id_separator = None, author_id = True, keyword_id = False, author_col_name = None, author_ids_col_name = None, keywords_col_name = None):
    
    if bibtex_or_csv == 'bibtex':
        df_pub = bibtex_to_df(pub_file_path)
    elif bibtex_or_csv == 'csv':
        df_pub = pd.read_csv(pub_file_path)
    else:
        print('Only support bibtex and csv')
    
    if database == 'scopus':
        print('Start the process for scopus')
        
        generate_author_files_with_defined_id_to_csv(df_pub, 'author_ids', 'author_names', output_dir, ';', ';')
        generate_keywords_files_without_defined_id_to_csv(df_pub, 'authkeywords', ' | ', output_dir)
        
    elif database == 'acm':
        print('Start the process for acm')
        generate_author_files_without_defined_id_to_csv(df_pub, 'author', ', ', output_dir)
        generate_keywords_files_without_defined_id_to_csv(df_pub, 'keywords', ', ', output_dir)
    
    #This function is the complex version for user choice
    elif user_choice == True and author_separator != None and keywords_separator != None and author_id_separator != None:
        print('Start the user choice')
        
        if keyword_provided == False:
            #Generate keywords from title and abstract
            print('The keywords is not provided and start to generate the keywords')
            generate_keywords(df_pub, 'title', 'abstract')
        if author_id == True and keyword_id == True:
            generate_author_files_with_defined_id_to_csv(df_pub, author_ids_col_name, author_col_name, output_dir, author_id_separator, author_separator)
            generate_keywords_files_with_defined_id_to_csv(df_pub, keywords_id_col_name, keywords_col_name, output_dir)
            
        elif author_id == False and keyword_id == False:
            generate_author_files_without_defined_id_to_csv(df_pub, author_ids_col_name, author_col_name, output_dir, author_id_separator, author_separator)
            generate_keywords_files_without_defined_id_to_csv(df_pub, keywords_col_name, keywords_separator, output_dir)

        elif author_id == True and keyword_id == False:
            generate_author_files_with_defined_id_to_csv(df_pub, author_ids_col_name, author_col_name, output_dir, author_id_separator, author_separator)
            generate_keywords_files_without_defined_id_to_csv(df_pub, keywords_col_name, keywords_separator, output_dir)

        elif author_id == False and keyword_id == True:
            generate_author_files_without_defined_id_to_csv(df_pub, author_ids_col_name, author_col_name, output_dir, author_id_separator, author_separator)
            generate_keywords_files_with_defined_id_to_csv(df_pub, keywords_id_col_name, keywords_col_name, output_dir)
            
    else:
        print('the database is not supported, but you can access the root functions.')
     
    print('Start to find the CRYS score')
    nodes_path = output_dir + '\researchers.csv'
    edges_path = output_dir + '\collabrations.csv'
    find_crys_score_from_scopus(pub_file_path, nodes_path, edges_path, output_dir, apikey_file, corpus_file_path)


# In[37]:


#This is the third choice that the user got all the data directly from scopus
def search_from_scopus(keywords, output_dir, apikey_file, countries_file, year, corpus_file_path, vector_size = 300, window_size = 3, sg = 0, hs = 1, epochs = 100, alpha = 0.001):
    
    #Step 1: Download the publications
    download_publications(keywords, output_dir, countries_file, year)
    #Step 2: Generate nodes and edges from the publications
    get_collaboration(publication_file_path, output_dir)#nodes and edges for authors
    generate_graph_files(publication_file_path, output_dir)#nodes and edges for keywords
    #Step 3: Clean the corpus file
    preprocess(corpus_file_path, output_dir)
    #Step 4: Generate the w2v model from the cleaned corpus file
    cleaned_corpus_file = output_dir + '\c_' + os.path.basename(corpus_file_path)
    create_model(cleaned_corpus_file, output_dir, vector_size, window_size, sg, hs, epochs, alpha)
    #Step 5: Generate vocabulary file
    get_vocabulary(cleaned_corpus_file, output_dir)
    #Step 6: Calculate the Field Networking Index
    model_file = output_dir + '\w2v.model'
    vocabulary_file = output_dir + '\vocabulary.csv'
    get_score(output_dir, model_file, vocabulary_file)
    #Step 7: Get the H-index, Field-weighted Citation Impact (FWCI), etc
    researchers_file = output_dir + '\researchers.csv'
    all_pubs_file = output_dir + '\all_publications.csv'
    metric_extraction(apikey_file, researchers_file, all_pubs_file)
    #Step 8: Classify the researchers on clusters
    authors_file = output_dir + '\researchers.csv'
    cluster_analysis(authors_file)
    
    #Step 9: Export pubs, nodes, and edges to JSON
    pub_file_name = "publications_{}_{}.csv".format(keywords_list, year)

    df_pub_file = pd.read_csv(output_dir + '\%s'%(pub_file_name), index=False, sep = '\t')
    df_author_nodes_file = pd.read_csv(output_dir + '\researchers.csv', index=False)
    df_author_edges_file = pd.read_csv(output_dir + '\collaboration.csv', index=False)
    df_keywords_nodes_file = pd.read_csv(output_dir + '\nodes.csv', index=False)
    df_keywords_edges_file = pd.read_csv(output_dir + '\edges.csv', index=False)
    
    df_pub_file.to_json(output_dir + '\nodes.json', orient="records", indent = 4)   
    export_two_df_to_one_json(df_author_nodes_file, df_author_edges_file, output_dir, 'authors')
    export_two_df_to_one_json(df_keywords_nodes_file, df_keywords_edges_file, output_dir, 'keywords')


# In[38]:


#This function is for the advanced usage of choice 2, which is read the bibtex file directly.
#This function could read the existed publication, nodes, and edges files in the local folder and process them to get more information of authors.
def find_crys_score_from_scopus(pub_path, output_dir, apikey_file, corpus_file_path, vector_size = 300, window_size = 3, sg = 0, hs = 1, epochs = 100, alpha = 0.001):
    
    #Step 1: Download the publications
    
    #Step 2: Generate nodes and edges from the publications
    
    
    #Step 3: Clean the corpus file
    preprocess(corpus_file_path, output_dir)
    #Step 4: Generate the w2v model from the cleaned corpus file
    cleaned_corpus_file = output_dir + '\c_' + os.path.basename(corpus_file_path)
    create_model(cleaned_corpus_file, output_dir, vector_size, window_size, sg, hs, epochs, alpha)
    #Step 5: Generate vocabulary file
    get_vocabulary(cleaned_corpus_file, output_dir)
    #Step 6: Calculate the Field Networking Index
    model_file = output_dir + '\w2v.model'
    vocabulary_file = output_dir + '\vocabulary.csv'
    get_score(output_dir, model_file, vocabulary_file)
    #Step 7: Get the H-index, Field-weighted Citation Impact (FWCI), etc
    researchers_file = output_dir + '\researchers.csv'
    all_pubs_file = output_dir + '\all_publications.csv'
    metric_extraction(apikey_file, researchers_file, all_pubs_file)
    #Step 8: Classify the researchers on clusters
    authors_file = output_dir + '\researchers.csv'
    cluster_analysis(authors_file)
    
    #Step 9: Export pubs, nodes, and edges to JSON
    pub_file_name = pub_path

    df_pub_file = pd.read_csv(output_dir + '\%s'%(pub_file_name), index=False)
    df_author_nodes_file = pd.read_csv(output_dir + '\researchers.csv', index=False)
    df_author_edges_file = pd.read_csv(output_dir + '\collaboration.csv', index=False)
    df_keywords_nodes_file = pd.read_csv(output_dir + '\nodes.csv', index=False)
    df_keywords_edges_file = pd.read_csv(output_dir + '\edges.csv', index=False)
    
    df_pub_file.to_json(output_dir + '\nodes.json', orient="records", indent = 4)   
    export_two_df_to_one_json(df_author_nodes_file, df_author_edges_file, output_dir, 'authors')
    export_two_df_to_one_json(df_keywords_nodes_file, df_keywords_edges_file, output_dir, 'keywords')
    


# In[39]:


def create_new_folder(path = ''):
    time = datetime.now()
    
    results_dir = path + '\search_results_%d_%d_%d_%d_%d_%d'%(time.year, time.month, time.day, time.hour, time.minute, time.second)
    
    os.mkdir(results_dir)
    
    print('the folder is created in the path.')
    return results_dir


# In[ ]:





# In[39]:


#Test Example of read bibtex or csv file

# file_path = r'C:\Users\Administrator\Desktop\Previous_document\user_data\wrapper_test_file_for_scopus.csv'
# output_dir = create_new_folder(r'C:\Users\Administrator\Desktop\Previous_document')
# 
# 
# # In[44]:
# 
# 
# output_dir
# 
# 
# # In[43]:
# 
# 
# generate_files_without_crys_score('csv', file_path, 'scopus', output_dir)
# 
# 
# # In[42]:
# 
# 
# #Test example for reading acm data in bitex format
# acm_bibtex_file_path = r'C:\Users\Administrator\Downloads\Science Religion_2012_2022_acm.bib'
# 
# 
# # In[43]:
# 
# 
# generate_files_without_crys_score('bibtex', acm_bibtex_file_path, 'acm', r'C:')
# 
# 
# # In[41]:
# 
# 
# #Test example of google scholar search
# google_scholar_search('google_scholar', 'Perception of physical stability and center of mass of 3D objects', '75907e416291e41bef91fddcebe9973b', [2000, 2022], r'C:')
# 

# In[ ]:




