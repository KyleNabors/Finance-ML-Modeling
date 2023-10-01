#Base Libraries
import os 
import sys
import json
import csv

#Core Libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pygwalker as pyg

#Model Libraries
from bertopic import BERTopic
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
#ML Libraries
import torch 

#Importing Configs
# Define the path where config.py is located
config_path = os.getcwd()
print(config_path)

# Add this path to the sys.path
sys.path.append(config_path)

# Now Python knows where to find config.py
import config

#Configs
database_file = config.database
database_folder = config.database_folder
bert_models = config.bert_models
bert_models_local = config.bert_models_local
Model_Folder = config.texts

Body = config.Body
Model = config.Model
Model_Subfolder = f'/{Body} Texts/{Model}'

Model_Folder = Model_Folder + Model_Subfolder

df = pd.read_csv(f"{Model_Folder}/{Model}_texts.csv")  
docs = df["segment"].to_list()
timestamps = df['date'].to_list()
type = df['type'].to_list()

Body_2 = config.Body_2
Model_2 = config.Model_2
Model_Subfolder_2 = f'/{Body_2} Texts/{Model_2}'
Model_Folder_2 = Model_Folder + Model_Subfolder_2

df_2 = pd.read_csv(f"{Model_Folder_2}/{Model_2}_texts.csv")  
docs_2 = df_2["segment"].to_list()
timestamps_2 = df_2['date'].to_list()
type_2 = df_2['type'].to_list()

topic_model = torch.load(f"{bert_models_local}/{Body}/{Model}/topic_model_{Model}.pt")
topic_model_2 = torch.load(f"{bert_models_local}/{Body_2}/{Model_2}/topic_model_{Model_2}.pt")

sim_matrix = cosine_similarity(topic_model.get_topics(), topic_model_2.get_topics())

