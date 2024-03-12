import os 
import sys
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import torch 
import nltk
import spacy
import tensorflow as tf

nltk.download('punkt')

import platform
platform.platform()

torch.backends.mps.is_built()

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")
    
test = "This is a sentence that we would find in a financial report."

print(len(test))


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

df = pd.read_csv(f"{Model_Folder}/{Model}_texts.csv") 
if Model == "Beige Book":
    print("skip")
else:
    df = df[df['language'] == 'en']



#Finbert 
# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

tokenizer_1 = AutoTokenizer.from_pretrained("ProsusAI/finbert", force_download=True)
model_1 = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", force_download=True)
model_1 = model_1.to('mps')

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',
                                                        num_labels=3, force_download=True)
finbert = finbert.to('mps')
tokenizer_2 = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone', force_download=True)

labels = {0:'positive', 1:'negative',2:'neutral'}
labels2 = {0:'neutral', 1:'positive',2:'negative'}
out_1= []
out_2 = []
sent_val = list()
tone_val = list()
long = 0
errors = 0
total = 0


for index, row in df.iterrows():
    docs = row["segment"]
    timestamps = row['date']
    title = row['title']
    docs = str(docs)
    doc_num = row['doc_num']
    
    total += 1
    try:
        inputs_1 = tokenizer_1(docs, return_tensors="pt", padding='max_length', max_length=511).to('mps')
        outputs_1 = model_1(**inputs_1)
        val_1 = torch.nn.functional.softmax(outputs_1.logits, dim=-1).to('cpu')
        val_1 = val_1.detach().numpy()  
        
        positive = val_1[:, 0][0]
        negative = val_1[:, 1][0]
        neutral = val_1[:, 2][0]
        net = labels[np.argmax(val_1)]

        out_1.append([doc_num, timestamps, title, docs, positive, negative, neutral, net])
        

        # inputs_2 = tokenizer_2(docs, return_tensors="pt", padding='max_length', max_length=511).to('mps')
        # outputs_2 = finbert(**inputs_2)[0]
        # val_2 = labels2[np.argmax(outputs_2.to('cpu').detach().numpy())]
        # out_2.append([doc_num, timestamps, title, docs, val_2])
        
    except:
        errors += 1

percent = (errors/total)*100
print(f'Errors Long: {errors}')
print(f'Errors Long %: {percent}')

df_out_1 = pd.DataFrame(out_1, columns=["doc_num", "date", "title", "segment", "positive", "negative", "neutral", "sentiment"])
df_out_1["sentiment"] = df_out_1["sentiment"].replace({'positive': 1, 'neutral' : 0, 'negative' : -1})
df_out_1.to_csv(f"{finbert_models}/{Body}/{Model}/{Body}_{Model}_finbert_model_short.csv")  

# df_out_2 = pd.DataFrame(out_2, columns=["doc_num", "date", "title", "segment", "tone"])
# df_out_2.to_csv(f"{finbert_models}/{Body}/{Model}/{Body}_{Model}_finbert_model_short_2.csv") 
print('done')