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
warnings.filterwarnings("ignore", category=DeprecationWarning)


#Preping Data
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


lemmatized_texts = lemmatization(data)
print (lemmatized_texts[0][0:90])

def gen_words(texts):
    final = []
    for text in texts:
        new = gensim.utils.simple_preprocess(text, deacc=True)
        final.append(new)
    return(final)

data_words = gen_words(lemmatized_texts)

print(data_words[0][0:20])

  

