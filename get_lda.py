import json
import os
import os.path as osp
import time
import numpy as np
from collections import OrderedDict

import sys
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
import gensim
from gensim import utils, corpora, models
from gensim.corpora.wikicorpus import filter_wiki
from gensim.parsing.preprocessing import STOPWORDS

import argparse

parser = argparse.ArgumentParser(description='get LDA topic labels')

parser.add_argument('--num_topics', type=int, default='40',
                    help='#topics of LDA')
parser.add_argument('--lda_dir', type=str, default='data',
                    help='lda dir')
parser.add_argument('--data_dir', type=str, default='data',
                    help='dir to save labels')

args = parser.parse_args()

def preprocess(raw):
    # Initialize Tokenizer
    tokenizer = RegexpTokenizer(r'\w+')

    # Initialize Lemmatizer
    lemma = WordNetLemmatizer()
    
    # Decode Wiki Markup entities and remove markup
    text = filter_wiki(raw)
    text = re.sub(filter_more, '', text)

    # clean and tokenize document string
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    
    # remove stop words from tokens
    tokens = [i for i in tokens if not i in STOPWORDS]

    # stem tokens
    tokens = [lemma.lemmatize(i) for i in tokens]

    # remove non alphabetic characters
    tokens = [re.sub(r'[^a-z]', '', i) for i in tokens]
    
    # remove unigrams and bigrams
    tokens = [i for i in tokens if len(i)>2]
    
    return tokens


filter_more = re.compile('(({\|)|(\|-)|(\|})|(\|)|(\!))(\s*\w+=((\".*?\")|([^ \t\n\r\f\v\|]+))\s*)+(({\|)|(\|-)|(\|})|(\|))?', re.UNICODE | re.DOTALL | re.MULTILINE) 

# tokenization
print('Tokenizing training data')
data = [json.loads(line) for line in open('data/train.vispro.pool.1.1.bert.512.jsonlines')]
texts = []
doc_keys = []
for dialog in data:
    raw = ''
    for sent in dialog['original_sentences'][1:]:
        # caption not included
        raw += ' '.join(sent)
        raw += ' '
    raw = raw[:-1]
    texts.append(preprocess(raw))
    doc_keys.append(dialog["doc_key"])

# construct dictionary
dictionary = corpora.Dictionary(texts)
dictionary_filename = osp.join(args.lda_dir, 'dictionary.dict')
dictionary.save(dictionary_filename)
dictionary.filter_extremes(no_below=20, no_above=0.5)
corpus = [dictionary.doc2bow(text) for text in texts]

# training model
print('Training LDA model')
start_time = time.time()
ldamodel = models.ldamodel.LdaModel(corpus, num_topics=args.num_topics, id2word = dictionary, passes=20)
print('LDA training finished. %.2fs passed' % (time.time() - start_time))
s = ldamodel.print_topics(-1)
print(s)

model_filename = osp.join(args.lda_dir, 'ldamodel'+str(args.num_topics)+'.lda')
ldamodel.save(model_filename)
print('LDA model saved to ' + model_filename)

# predict topics for each document
labels = OrderedDict()
for split in ['train', 'val', 'test']:
    print('Processing %s data' % split)
    if split in ['val', 'test']:
        data = [json.loads(line) for line in open('data/%s.vispro.pool.1.1.bert.512.jsonlines' % split)]
        texts = []
        doc_keys = []
        for dialog in data:
            raw = ''
            for sent in dialog['original_sentences'][1:]:
                raw += ' '.join(sent)
                raw += ' '
            raw = raw[:-1]
            texts.append(preprocess(raw))
            doc_keys.append(dialog['doc_key'])
        corpus = [dictionary.doc2bow(text) for text in texts]
    for i, (document, doc_key) in enumerate(zip(corpus, doc_keys)):
        lda_vector = ldamodel.get_document_topics(corpus[i], minimum_probability=None)
        topic_label = np.zeros(args.num_topics)
        for topic, prob in lda_vector:
            topic_label[topic] = prob
        labels[doc_key] = topic_label

# save result
output_file = osp.join(args.data_dir, 'lda%dlabels.npz' % args.num_topics)
np.savez(output_file, labels=labels)
print('Results saved to ' + output_file)