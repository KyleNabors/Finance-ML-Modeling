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

#ML Libraries
import torch 

#Importing Configs
# Define the path where config.py is located
config_path = os.getcwd()
print(config_path)
exit()
# Add this path to the sys.path
sys.path.append(config_path)

# Now Python knows where to find config.py
import config

#Configs
database_file = config.database
database_folder = config.database_folder
bert_models = config.bert_models
bert_models_local = config.bert_models_local


df = pd.read_csv("/Users/kylenabors/Documents/MS-Thesis Data/Database/Fed Data/fed_data_blocks.csv")
docs = df["segment"].to_list()
timestamps = df['date'].to_list()
type = df['type'].to_list()

#BERT Models
topic_model_BB = torch.load(f"{bert_models_local}/topic_model_fed_BB.pt")
topic_model_all = torch.load(f"{bert_models_local}/topic_model_fed_all.pt")

#topics_over_time = topic_model_BB.topics_over_time(docs, timestamps, nr_bins=100)

#df_tot = pd.DataFrame(topics_over_time, columns=['Topic', 'Words', 'Frequency', 'Timestamp'])
#df_tot.to_csv(f"{bert_models_local}/tot.csv", index=True)

df_tot = pd.read_csv(f"{bert_models_local}/tot.csv")

pivot_df_tot = df_tot.pivot(index='Timestamp', columns='Topic', values='Frequency')

gwalker = pyg.walk(pivot_df_tot)




exit()
#save topics over time graph as HTML file
topic_model_BB.visualize_topics_over_time(topics_over_time, top_n_topics=8).write_html(f"{bert_models}/topics_over_time.html")










#Create Embedding
sentence_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

#Reduce Dimensionality
umap_model = UMAP(n_neighbors=15, 
                  n_components=2, 
                  min_dist=0.0, 
                  metric='cosine', 
                  #target_metric_kwds=keywords, 
                  target_weight=0.5, 
                  verbose=True)

sim_matrix = cosine_similarity(topic_model_BB.topic_embeddings_, topic_model_all.topic_embeddings_)

topic = 10
most_similar_topic = np.argmax(sim_matrix[topic + 1])-1
print(topic_model_BB.get_topic(most_similar_topic))

#topic_model.visualize_barchart(top_n_topics = 100, n_words=8).write_html(f"{bert_models}/barchart.html")