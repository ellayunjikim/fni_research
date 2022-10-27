import glob
import os
import re
import string
import sys

import pandas as pd
from nltk import corpus
from nltk import FreqDist
from nltk.tokenize import word_tokenize
import numpy as np
import spacy

HELP_MESSAGE = 'Usage:\npython {} --corpus=<input_file> --output=<output_dir>'
INVALID_ARG_NUMBER = 'Expected 2 argument, but {} received.'
INSTRUCTIONS_MESSAGE = 'corpus: corpus file (each line representing a paragraph, header="text")'

def normalize_case(data):
    data.text = data.text.apply(str.lower)

def tokenize(data):
    data.text = data.text.apply(lambda text: word_tokenize(text, language='spanish'))

def remove_punctuations(data):
    trans_table = str.maketrans('', '', string.punctuation)
    remove_punctuations = lambda token: token.translate(trans_table)
    data.text = data.text.apply(lambda tokens: list(map(remove_punctuations, tokens)))

def remove_numerics(data):
    data.text = data.text.apply(lambda tokens: list(filter(str.isalpha, tokens)))

def remove_short_words(data, short_word_length=2):
    not_short_word = lambda token: len(token) > short_word_length and len(token) < 31
    data.text = data.text.apply(lambda tokens: list(filter(not_short_word, tokens)))

def remove_stopwords(data, stopwords):
    not_stopword = lambda token: token not in stopwords
    data.text = data.text.apply(lambda tokens: list(filter(not_stopword, tokens)))

def remove_short_lines(data, short_line_words=0):
    return data[data.text.str.len() > short_line_words]

def undo_tokenization(data):
    data.text = data.text.apply(' '.join)
    
def lemmatize(data, nlp):
    data.text = data.text.apply(lambda token: [y.lemma_ for y in nlp(token)])

def remove_rare_words(data):
    tokens = []
    for text in data.text:
        tokens.extend(text)
    frequencies = FreqDist(tokens)
    rare_words = filter(lambda pair: pair[1] < 3, frequencies.items())
    rare_words = set(map(lambda pair: pair[0], rare_words))
    not_rare_word = lambda token: token not in rare_words
    data.text = data.text.apply(lambda tokens: list(filter(not_rare_word, tokens)))

def remove_repeated_or_similar(data):
    return data.drop_duplicates(subset=['text'])

def preprocess(input_path, output_dir):
    data = pd.read_csv(input_path, engine='python', sep='\t\t')
    data = data[~data.text.isna()]
    nlp = spacy.load("en_core_web_sm")
    normalize_case(data)
    tokenize(data)
    remove_punctuations(data)
    remove_numerics(data)
    remove_short_words(data, short_word_length=2)
    remove_stopwords(data, corpus.stopwords.words('english'))
    undo_tokenization(data)
    data = remove_repeated_or_similar(data)
    lemmatize(data, nlp)
    remove_rare_words(data)
    data = remove_short_lines(data, short_line_words=5)
    undo_tokenization(data)

    input_dir, input_file = os.path.split(input_path)
    output_path = os.path.join(output_dir, f'c_{input_file}')
    data.to_csv(output_path, index=False)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(INVALID_ARG_NUMBER.format(len(sys.argv) - 1))
        print(HELP_MESSAGE.format(sys.argv[0]))
        print(INSTRUCTIONS_MESSAGE)
        exit(1)
    
    input_file = ''
    output_dir = ''
    
    for arg in sys.argv[1:]:
        if arg.startswith('--corpus='): input_file = arg.split('=')[-1]
        elif arg.startswith('--output=') : output_dir = arg.split('=')[-1]
    preprocess(input_file, output_dir)
