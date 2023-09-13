import os
import sys
import csv
import json
import pandas as pd
import pygwalker as pyg

# Load data from the JSON database
def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# Write data to a JSON file
def write_data(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        
#Find and import config file
config_path = os.getcwd()
sys.path.append(config_path)
import config

#Variables, Paramaters, and Pathnames needed for this script
keyword_freq_ts = config.keyword_freq_ts
keyword_freq_ts = load_data(keyword_freq_ts)



df = pd.DataFrame(keyword_freq_ts)
print(df.head())

#gwalker = pyg.walk(keyword_freq_ts)