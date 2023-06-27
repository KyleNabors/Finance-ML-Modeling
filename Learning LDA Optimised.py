import os
import nltk
import ssl
import numpy as np
import json
import glob
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy
from nltk.corpus import stopwords
import pyLDAvis
import pyLDAvis.gensim
import warnings

nltk.download("stopwords")

def load_data(file):
    with open(file) as f:
        data = json.load(f)
        return data

def write_data(file, data):
    with open(file, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4)

stopwords = stopwords.words('english')
print(stopwords)
data = load_data("/Users/kylenabors/Documents/GitHub/MS-Thesis/Training Data/ushmm_dn.json")["texts"]
print(data[0][0:90])

# Disable other components
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]

def lemmatization(texts, allowed_postags=allowed_postags):
    docs = nlp.pipe(texts)
    lemmatized_texts = []
    for doc in docs:
        lemmatized_texts.append(" ".join([token.lemma_ for token in doc if token.pos_ in allowed_postags]))
    return lemmatized_texts

lemmatized_texts = lemmatization(data)
print (lemmatized_texts[0][0:90])

def gen_words(texts):
    return [gensim.utils.simple_preprocess(text, deacc=True) for text in texts]

data_words = gen_words(lemmatized_texts)
print(data_words[0][0:20])
id2word = corpora.Dictionary(data_words)

# Using a generator for memory-efficient corpus creation
def create_corpus(data_words):
    for text in data_words:
        yield id2word.doc2bow(text)

corpus = list(create_corpus(data_words))
print(corpus[0][0:20])

word = id2word[[0][:1][0]]
print(word)

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=30,
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha='auto')

vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, mds='mmds', R=30)


