import os 
import sys
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import torch 
import nltk
import spacy

#Importing Configs
# Define the path where config.py is located
#Mac
os.chdir('/Users/kylenabors/Documents/GitHub/Finance-ML-Modeling')
#Linux
#os.chdir('/home/kwnabors/Documents/GitHub/Finance-ML-Modeling')
config_file_path = os.getcwd()
print(config_file_path)

# Add this path to the sys.path
sys.path.append(config_file_path)

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
df_2 = pd.read_csv(f"{Model_Folder}/{Model}_texts.csv")  
df_2 = df_2[df_2['language'] == 'en']

#Finbert 
# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

tokenizer_1 = AutoTokenizer.    ("ProsusAI/finbert")
model_1 = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer_2 = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer_2)

labels = {0:'neutral', 1:'positive',2:'negative'}
out_1= []
out_2 = []
sent_val = list()
tone_val = list()
long = 0
errors = 0

for index, row in df_2.iterrows():
    docs = row["segment"]
    timestamps = row['date']
    type = row['type']
    title = row['title']
    docs = str(docs)
    doc_num = row['doc_num']
    
    try:
            results = nlp(docs)
    except:
            errors += 1
            continue
    
    results = results[0]['label']
    # if results == "Negative":
    #     r_num = -1 
    # if results == "Neutral":
    #     r_num = 0
    # if results == "Positive":
    #     r_num = 1

    out_1.append([doc_num, timestamps, title, results, docs])
    
df_out_1 = pd.DataFrame(out_1, columns=["doc_num", "date", "title", "sentiment", "segment"])
df_out_1_2 = df_out_1[["doc_num", "sentiment"]]
df_out_1_2 = df_out_1_2.groupby(['doc_num']).mean()

print(df_out_1.head())

print(f'The analysis failed {errors} times.')


for index, row in df.iterrows():
    docs = row["segment"]
    timestamps = row['date']
    #type = row['type']
    title = row['title']
    docs = str(docs)
    doc_num = row['doc_num']
    
    inputs_2 = tokenizer_2(docs, return_tensors="pt", padding=True, truncation=True, max_length=511)
    outputs_2 = finbert(**inputs_2)[0]
    val_2 = labels[np.argmax(outputs_2.detach().numpy())]
    #tone_val.append(val_2)

    out_2.append([doc_num, timestamps, title, type, docs, val_2])

df_out_2 = pd.DataFrame(out_2, columns=["doc_num", "date", "title", "type", "segment", "tone"])


df_out = df_out_2.merge(df_out_1_2, on='doc_num', how='inner')

df_out.to_csv(f"{finbert_models}/{Body}_{Model}_finbert model.csv")  
df_out_1.to_csv(f"{finbert_models}/{Body}_{Model}_finbert model_line.csv")  

print('done')