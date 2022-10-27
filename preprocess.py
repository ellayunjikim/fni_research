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

def normalize_case(text):
    return text.lower()

def tokenize(text):
    return word_tokenize(text, language='english')

def remove_punctuations(text):
    trans_table = str.maketrans('', '', string.punctuation)
    remove_punctuations = lambda token: token.translate(trans_table)
    return list(map(remove_punctuations, text))

def remove_numerics(text):
    return list(filter(str.isalpha, text))

def remove_short_words(text, short_word_length=2):
    not_short_word = lambda token: len(token) > short_word_length and len(token) < 31
    return list(filter(not_short_word, text))

def remove_stopwords(text, stopwords):
    not_stopword = lambda token: token not in stopwords
    return list(filter(not_stopword, text))

def undo_tokenization(text):
    return ' '.join(text)

def remove_accents(text):
    accents_map = { 'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u' }
    for with_accent, no_accent in accents_map.items():
        text = text.replace(with_accent, no_accent)
    return text

def lemmatize(text, nlp):
    return [y.lemma_ for y in nlp(text)]

def preprocess(text):
    # DO NOT change the order of this calls unless you REALLY know what you are doing
    # Changing the order can cause runtime errors or (WORSE) no errors but poor results
    nlp = spacy.load("en_core_web_sm")
    text = normalize_case(text)
    text = remove_accents(text)
    text = tokenize(text)
    text = remove_punctuations(text)
    text = remove_numerics(text)
    text = remove_short_words(text, short_word_length=2)
    text = undo_tokenization(text) # Need to be string for lemmatization
    text = lemmatize(text, nlp)
    text = remove_stopwords(text, corpus.stopwords.words('english'))
    text = undo_tokenization(text) # Need to be string for remove_repeated_or_similar to work
    text = tokenize(text) # Need to be tokens for remove_rare_words and remove_short_lines to work
    return text