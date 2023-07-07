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
import matplotlib.pyplot as plt
import pandas as pd
warnings.filterwarnings("ignore",category=DeprecationWarning)

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
data = load_data("/Users/kylenabors/Documents/GitHub/MS-Thesis/Database/Fed Data/fed_data_interest_train.json")
print(data[0][0:90])

#Remove Stopwords
def lemmatization(data, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    texts_out = []
    for text, _ in data:  # unpack the sublist into text and label
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

#Bigrams and Trigrams
bigrams_phrases = gensim.models.Phrases(data_words, min_count=5, threshold=100)
trigrams_phrases = gensim.models.Phrases(bigrams_phrases[data_words], threshold=100)

bigrams = gensim.models.phrases.Phraser(bigrams_phrases)
trigram = gensim.models.phrases.Phraser(trigrams_phrases)

def make_bigrams(texts):
    return([bigrams[doc] for doc in texts])

def make_trigram(texts):
    return([trigram[bigrams[doc]] for doc in texts])

data_bigrams = make_bigrams(data_words)
data_bigrams_trigrams = make_trigram(data_bigrams)

print(data_bigrams_trigrams[0])

#TF-IDF REMOVAL
from gensim.models import TfidfModel

id2word = corpora.Dictionary(data_bigrams_trigrams)

texts = data_bigrams_trigrams

corpus = [id2word.doc2bow(text) for text in texts]
# print (corpus[0][0:20])

tfidf = TfidfModel(corpus, id2word=id2word)

low_value = 0.03
words  = []
words_missing_in_tfidf = []
for i in range(0, len(corpus)):
    bow = corpus[i]
    low_value_words = [] #reinitialize to be safe. You can skip this.
    tfidf_ids = [id for id, value in tfidf[bow]]
    bow_ids = [id for id, value in bow]
    low_value_words = [id for id, value in tfidf[bow] if value < low_value]
    drops = low_value_words+words_missing_in_tfidf
    for item in drops:
        words.append(id2word[item])
    words_missing_in_tfidf = [id for id in bow_ids if id not in tfidf_ids] # The words with tf-idf socre 0 will be missing

    new_bow = [b for b in bow if b[0] not in low_value_words and b[0] not in words_missing_in_tfidf]
    corpus[i] = new_bow
    
    
    #Create Corpus
#id2word = corpora.Dictionary(data_words)

#corpus = []
#for text in data_words:
#    new = id2word.doc2bow(text)
#    corpus.append(new)
#print(corpus[0][0:20])

#word = id2word[[0][:1][0]]
#print(word)
    
    
 #Build LDA Model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus[:-1],
                                           id2word=id2word,
                                           num_topics=30,
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha="auto")   
    
    
test_doc = corpus[-1]

vector = lda_model[test_doc]
print (vector)

def Sort(sub_li):
    sub_li.sort(key = lambda x: x[1])
    sub_li.reverse()
    return (sub_li)
new_vector = Sort(vector)
print (new_vector)

lda_model.save("/Users/kylenabors/Documents/GitHub/MS-Thesis/Models/test_model.model")

new_model = gensim.models.ldamodel.LdaModel.load("/Users/kylenabors/Documents/GitHub/MS-Thesis/Models/test_model.model")

test_doc = corpus[-1]

vector = new_model[test_doc]
print (vector)

def Sort(sub_li):
    sub_li.sort(key = lambda x: x[1])
    sub_li.reverse()
    return (sub_li)
new_vector = Sort(vector)
print (new_vector)

                                            
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, mds='mmds', R=30)
pyLDAvis.save_html(vis, "/Users/kylenabors/Documents/GitHub/MS-Thesis/Models/LDA Test.html")