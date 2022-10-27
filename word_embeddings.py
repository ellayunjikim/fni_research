import sys
import os

import pandas as pd
from gensim import models
import nltk

HELP_MESSAGE = 'Usage:\npython {} --corpus=<input_file> --output=<output_dir> optional flags: --vector --window --sg --hs --epochs --alpha'
INVALID_ARG_NUMBER = 'Expected 2 argument, but {} received.'
INSTRUCTIONS_MESSAGE = 'corpus: corpus file (each line representing a paragraph, header="text")\nCheck https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec for more details on each optional flag'

VECTOR_SIZE = 300
WINDOW_SIZE = 3
SG = 0
HS = 1
EPOCHS = 100
ALPHA = 0.001

MODEL_FILE = 'w2v.model'

def create_model(corpus_file, output_dir, vector_size, window_size, sg, hs, epochs, alpha):
    corpus = pd.read_csv(corpus_file)
    sentences = list(corpus['text'])
    model_documents = [nltk.word_tokenize(sent) for sent in sentences]
    model = models.Word2Vec(model_documents, seed=123, min_count=1, size=vector_size, window=window_size, sg=sg, hs=hs, alpha=alpha, iter=epochs)
    model.save(os.path.join(output_dir, f'{MODEL_FILE}'))

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(INVALID_ARG_NUMBER.format(len(sys.argv) - 1))
        print(HELP_MESSAGE.format(sys.argv[0]))
        print(INSTRUCTIONS_MESSAGE)
        exit(1)
        
    input_file = ''
    output_dir = ''
    vs = VECTOR_SIZE
    ws = WINDOW_SIZE
    sg = SG
    hs = HS
    e = EPOCHS
    a = ALPHA
    
    for arg in sys.argv[1:]:
        if arg.startswith('--corpus='): input_file = arg.split('=')[-1]
        elif arg.startswith('--output=') : output_dir = arg.split('=')[-1]
        elif arg.startswith('--vector=') : vs = arg.split('=')[-1]
        elif arg.startswith('--window=') : ws = arg.split('=')[-1]
        elif arg.startswith('--sg=') : sg = arg.split('=')[-1]
        elif arg.startswith('--hs=') : hs = arg.split('=')[-1]
        elif arg.startswith('--epochs=') : e = arg.split('=')[-1]
        elif arg.startswith('--alpha=') : a = arg.split('=')[-1]
        
    create_model(input_file, output_dir, vs, ws, sg, hs, e, a)