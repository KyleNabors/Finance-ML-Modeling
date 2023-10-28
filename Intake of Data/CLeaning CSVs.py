# Import necessary libraries
import sys
import os
import json
import csv
from collections import defaultdict, Counter
import spacy
import nltk
from nltk.corpus import words, stopwords
import pandas as pd
import re
import matplotlib.pyplot as plt
import gensim
from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize
import time

#Json read and write functions
def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def write_data(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

nlp = spacy.load("en_core_web_lg")
nltk.download("stopwords") 
stopwords_en = nltk.corpus.stopwords.words('english')
stopwords_sp = nltk.corpus.stopwords.words('spanish')
stopwords_fr = nltk.corpus.stopwords.words('french')
stopwords_it = nltk.corpus.stopwords.words('italian')
stopwords = stopwords_en + stopwords_sp + stopwords_fr + stopwords_it

nltk.download('words')
hyphenated_words = set(word for word in words.words() if '-' in word)
all_words = nltk.corpus.words.words('en')
#all_words = set(word for word in words.words())

#Find and import config file
config_path = os.getcwd()
sys.path.append(config_path)
import config

#Variables, Paramaters, and Pathnames needed for this script
database_file = config.database
database_folder = config.database_folder
Model_Folder = config.texts

database_data = load_data(database_file)

#Load Model Parameters
Body = config.Body
Model = config.Model
accepted_types = config.accepted_types
Model_Subfolder = f'/{Body} Texts/{Model}'
Model_Folder = Model_Folder + Model_Subfolder


#Fed Speeches 
if Body == "Fed":
    files = pd.read_csv(f"/Users/kylenabors/Documents/Database/Training Data/Fed/Speeches/fed_speeches_1995_2023.csv", encoding='UTF-8')

#ECB Speeches 
if Body == "ECB":
    files = pd.read_csv('/Users/kylenabors/Documents/Database/Training Data/ECB/Speeches/all_ECB_speeches.csv', sep = "|", encoding = "UTF-8")
    files['contents'] = files['contents'].astype(str)
    
    
print(files.head())
print(files.columns)
print(type(files))
print(files.describe())

def detect_language_with_langdetect(line): 
    from langdetect import detect_langs
    try: 
        langs = detect_langs(line) 
        for item in langs: 
            # The first one returned is usually the one that has the highest probability
            return item.lang, item.prob 
    except: return "err", 0.0 
    
for index, row in files.iterrows():
    
    if Body == "Fed":
        year_month_day = row['date']
        doc_type = row['speaker']
        text = row['text']
        title = row['title']
        doc_year = row['year']
        
    if Body == "ECB":
        year_month_day = row['date']
        doc_type = row['speakers']
        text = row['contents']
        title  = row['title']

    language, lang_prob = detect_language_with_langdetect(text)
    print(language, lang_prob)