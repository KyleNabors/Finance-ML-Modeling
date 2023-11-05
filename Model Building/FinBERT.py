import os 
import sys
import pandas as pd
import numpy as np

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, TextGeneration
from bertopic.vectorizers import ClassTfidfTransformer
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
import torch 
import nltk
import spacy

# NLTK English stopwords
nlp = spacy.load("en_core_web_lg")
nltk.download('stopwords') 
stopwords_en = nltk.corpus.stopwords.words('english')
stopwords_sp = nltk.corpus.stopwords.words('spanish')
stopwords_fr = nltk.corpus.stopwords.words('french')
stopwords_it = nltk.corpus.stopwords.words('italian')
stopwords_gr = nltk.corpus.stopwords.words('german')
stopwords = stopwords_en + stopwords_sp + stopwords_fr + stopwords_it + stopwords_gr


#Find and import config file
config_path = os.getcwd()
sys.path.append(config_path)
import config

#Variables, Paramaters, and Pathnames needed for this script
database_file = config.database
database_folder = config.database_folder
bert_models = config.bert_models
bert_models_local = config.bert_models_local
keywords = config.keywords
finbert_models = config.finbert_models

Body = config.Body
Model = config.Model
Model_Subfolder = f'/{Body} Texts/{Model}'
Model_Folder = config.texts
Model_Folder = Model_Folder + Model_Subfolder

df = pd.read_csv(f"{Model_Folder}/{Model}_texts_long.csv")  
df = df[df['language'] == 'en']
#Finbert 
# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

tokenizer_1 = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model_1 = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer_2 = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer_2)

labels = {0:'neutral', 1:'positive',2:'negative'}
out = []
sent_val = list()
tone_val = list()
long = 0
for index, row in df.iterrows():
    docs = row["segment"]
    timestamps = row['date']
    type = row['type']
    title = row['title']
    docs = str(docs)
    
    inputs_1 = tokenizer_1(docs, return_tensors="pt", padding=True, truncation=True, max_length=511)
    outputs = model_1(**inputs_1)[0]
    
    
    val = labels[np.argmax(outputs.detach().numpy())]
    num = np.argmax(outputs.detach().numpy())
    sent_val.append(val)
    
    
    
    inputs_2 = tokenizer_2(docs, return_tensors="pt", padding=True, truncation=True, max_length=511)
    outputs_2 = finbert(**inputs_2)[0]
    val_2 = labels[np.argmax(outputs_2.detach().numpy())]
    #tone_val.append(val_2)
    
    
    
    # segments = sent_tokenize(docs, language='english')
    # results = []
    # for segment in segments:
    #     if len(segment) < 1800:
    #         long = max(long, len(segment))
    #         result = nlp(segment)
    #         if result[0]['label'] == 'Positive':
    #             m = 1
    #         elif result[0]['label'] == 'Negative':
    #             m = -1
    #         else:
    #             m = 0
    #         results.append(m)
    
    # mean_value = sum(results) / len(results) 

    out.append([timestamps, title, type, docs, val, val_2])
df_out = pd.DataFrame(out, columns=["date", "title", "type", "segment", "sentiment", "tone"])
print(df_out.head())

df_out.to_csv(f"{finbert_models}/{Body}_{Model}_finbert model.csv")  








# df = pd.read_csv(f"{Model_Folder}/{Model}_texts.csv")  
# df = df[df['language'] == 'en']
# #Finbert 
# # Load model directly
# # from transformers import AutoTokenizer, AutoModelForSequenceClassification

# # tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
# # model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

# from transformers import BertTokenizer, BertForSequenceClassification

# finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
# tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

# labels = {0:'neutral', 1:'positive',2:'negative'}
# out = []
# sent_val = list()
# for index, row in df.iterrows():
#     docs = row["segment"]
#     timestamps = row['date']
#     type = row['type']
#     title = row['title']
#     docs = str(docs)

#     inputs = tokenizer(docs, return_tensors="pt", padding=True, truncation=True, max_length=511)
#     outputs = finbert(**inputs)[0]

#     val = labels[np.argmax(outputs.detach().numpy())]  
#     sent_val.append(val)
#     out.append([timestamps, title, type, docs, val])