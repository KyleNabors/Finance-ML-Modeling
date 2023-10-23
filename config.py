#Import Libraries
import os
import os.path
import sys
import pandas as pd
from pandas.plotting import table 
import json

# Load data from the JSON database
def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# Write data to a JSON file
def write_data(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

#User Filesystem 
config_path = os.getcwd()
print(config_path)

#subfolders = [f.path for f in os.scandir(config_path) if f.is_dir()]
#print(subfolders)

#print(os.path.abspath(os.curdir))
os.chdir("../..")
system = os.path.abspath(os.curdir)
print(system)
print(config_path)

#Paramaters 
range_1s = "2007-12-01"
range_1e = "2009-06-30"
range_1 = (range_1s, range_1e)
range_2s = "2009-07-01"
range_2e = "2019-12-31"
range_2 = (range_2s, range_2e)
range_3s = "2020-01-01"
range_3e = "2022-05-31"
range_3 = (range_3s, range_3e)
range_4s = "2022-06-01"
range_4e = "2023-12-31"
range_4 = (range_4s, range_4e)
year_ranges = [range_1, range_2, range_3, range_4]
keywords = ["interest", "inflation", "unemployment", "market", "bank"]
negative_words = ["decrease", "unemployment", "crisis", "negative", "downward"]

#Subfolders
models = f"{config_path}/Models"
fed_models = f"{config_path}/Models/Fed Models"
fed_funds_folder = f"{config_path}/Models/Fed Models/Graphs/Fed Funds"
sp500_folder = f"{config_path}/Models/Fed Models/Graphs/SP500"
sp500_change_folder = f"{config_path}/Models/Fed Models/Graphs/SP500 Change"
four_model_graph_folders = [f"{config_path}/Models/Fed Models/Four Models/Graphs/Period {i}" for i in range(1, 5)]
four_models_datapath = f"{system}/MS-Thesis Data/Database/Fed Data/Four Models"
four_models_models_folder = f"{config_path}/Models/Fed Models/Four Models"
database_folder = f"{system}/MS-Thesis Data/Database"
bert_models = f"{system}/Database/Models/BERT Models/Visuals"
bert_models_local = f"{system}/Database/Models/BERT Models"
texts = f"{system}/Database/Models/Texts"
Local_Database = f'{system}/Database'
Word2Vec_models = f'{system}/Database/Models/Word2Vec Models'
Sentiment_models = f'{system}/Database/Models/Sentiment Analysis Models'

#Files
database = f"{system}/MS-Thesis Data/Database/fed_database.json"
texts_json = f"{system}/MS-Thesis Data/Database/Fed Data/fed_data_blocks.json"
keyword_freq_ts =  f"{system}/MS-Thesis Data/Database/Fed Data/keyword_freq_ts_blocks.json"
fed_funds = pd.read_excel(f'{system}/MS-Thesis Data/Database/Fed Data/FedFundsRate.xlsx', sheet_name='Monthly')
sp500 = pd.read_csv(f'{system}/MS-Thesis Data/Database/Market Data/GSPC.csv')
keyword_info_ts = f'{models}/Word2Vec Models/keyword_info_ts.csv'

#Varaibles
scale = 1

Body = 'ECB'
Model = 'Speeches'
accepted_types = ['Speeches',
                  ]

Body_2 = 'Fed'
Model_2 = 'Speeches'
accepted_types_2 = ['Speeches',
                  ]

#'Fed'
#'Beige and Monetary'
#'Beige Book'
#'Monetary Policy Report'

#'Fed'
#'Speeches'
#'Speeches'

#'Fed'
#'Beige Book'
#'Beige Book'

#'Fed'
#'Monetary Policy Report'
#'Monetary Policy Report'

#'Fed'
#'Press Confrences'
#'Press Confrences'

#'ECB'
#'Press Confrences'
#'Press Confrences'

#'IMF'
#'Financial Stability Reports'
#'Financial Stability Reports'

#'Private Banks'
#'Annual Report Banks
#'JP Morgan Annual Reports',
#'Citi Annual Reports'