#Import Libraries
import os
import os.path
import sys
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
        

#Paramaters 
year_ranges = [(2006, 2007), (2008, 2009), (2010, 2019), (2020, 2023)]
keywords = ["interest", "inflation", "invest", "trade", "uncertain"]

#Subfolders
fed_models = "/Users/kylenabors/Documents/GitHub/MS-Thesis/Models/Fed Models"
fed_funds_folder = "/Users/kylenabors/Documents/GitHub/MS-Thesis/Models/Fed Models/Graphs/Fed Funds"
sp500_folder = "/Users/kylenabors/Documents/GitHub/MS-Thesis/Models/Fed Models/Graphs/SP500"
sp500_change_folder = "/Users/kylenabors/Documents/GitHub/MS-Thesis/Models/Fed Models/Graphs/SP500 Change"
four_model_graph_folders = [f"/Users/kylenabors/Documents/GitHub/MS-Thesis/Models/Fed Models/Four Models/Graphs/Period {i}" for i in range(1, 5)]
four_models_datapath = "/Users/kylenabors/Documents/MS-Thesis Data/Database/Fed Data/Four Models"
four_models_models_folder = "/Users/kylenabors/Documents/GitHub/MS-Thesis/Models/Fed Models/Four Models"

#Files
database = "/Users/kylenabors/Documents/MS-Thesis Data/Database/fed_database.json"

if os.path.exists('/Users/kylenabors/Documents/MS-Thesis Data/Database/Fed Data/keyword_info_ts.csv'):
    df_keyword_info_ts = pd.read_csv('/Users/kylenabors/Documents/MS-Thesis Data/Database/Fed Data/keyword_info_ts.csv')
else:
    df_keyword_info_ts = []
    
if os.path.exists('/Users/kylenabors/Documents/MS-Thesis Data/Database/Fed Data/keyword_freq_ts.json'):
    keyword_freq_ts = load_data("/Users/kylenabors/Documents/MS-Thesis Data/Database/Fed Data/keyword_freq_ts.json")
else:
    keyword_freq_ts = []
    
fed_funds = pd.read_excel('/Users/kylenabors/Documents/MS-Thesis Data/Database/Fed Data/FedFundsRate.xlsx', sheet_name='Monthly')
sp500 = pd.read_csv('/Users/kylenabors/Documents/MS-Thesis Data/Database/Market Data/Monthly SP.csv')

#Varaibles
