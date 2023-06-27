import os
import nltk

#Open Download Editor
#nltk.download()

import ssl
nltk.download("stopwords") 
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

#Json Functions
def load_data(file):
    with open(file) as f:
        data = json.load(f)
        return data
    
def write_data(file, data):
    with open(file, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4)
#Load Data
stopwords = stopwords.words('english')
print(stopwords)
data = load_data("/Users/kylenabors/Documents/GitHub/MS-Thesis/Training Data/ushmm_dn.json")["texts"]
print(data[0][0:90])

#Remove Stopwords
def lemmatization(texts, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    texts_out = []
    for text in texts:
        doc = nlp(text)
        new_text = []
        for token in doc:
            if token.pos_ in allowed_postags:
                new_text.append(token.lemma_)
        final = " ".join(new_text)
        texts_out.append(final)
    return (texts_out)

#Lemmatize
lemmatized_texts = lemmatization(data)
print (lemmatized_texts[0][0:90])

#Remove Stopwords
def gen_words(texts):
    final = []
    for text in texts:
        new = gensim.utils.simple_preprocess(text, deacc=True)
        final.append(new)
    return(final)

#Create Dictionary
data_words = gen_words(lemmatized_texts)
print(data_words[0][0:20])

#Create Corpus
id2word = corpora.Dictionary(data_words)

corpus = []
for text in data_words:
    new = id2word.doc2bow(text)
    corpus.append(new)

print(corpus[0][0:20])

word = id2word[[0][:1][0]]
print(word)

pyLDAvis.enable_notebook()

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=30,
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha='auto')
                                            
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, mds='mmds', R=30)
pyLDAvis.save_html(vis, "/Users/kylenabors/Documents/GitHub/MS-Thesis/Visualisations/USHMM DN.html")