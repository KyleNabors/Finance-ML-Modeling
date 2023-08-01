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

#User Filesystem 
config_path = os.getcwd()
print(os.path.abspath(os.curdir))
os.chdir("../..")
system = os.path.abspath(os.curdir)
print(system)

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
keywords = ["interest", "inflation", "invest", "trade", "uncertain"]

#Subfolders
fed_models = f"{system}/GitHub/MS-Thesis/Models/Fed Models"
fed_funds_folder = f"{system}/GitHub/MS-Thesis/Models/Fed Models/Graphs/Fed Funds"
sp500_folder = f"{system}/GitHub/MS-Thesis/Models/Fed Models/Graphs/SP500"
sp500_change_folder = f"{system}/GitHub/MS-Thesis/Models/Fed Models/Graphs/SP500 Change"
four_model_graph_folders = [f"{system}/GitHub/MS-Thesis/Models/Fed Models/Four Models/Graphs/Period {i}" for i in range(1, 5)]
four_models_datapath = f"{system}/MS-Thesis Data/Database/Fed Data/Four Models"
four_models_models_folder = f"{system}/GitHub/MS-Thesis/Models/Fed Models/Four Models"
database_file = f"{system}/MS-Thesis Data/Database"

#Files
database = f"{system}/MS-Thesis Data/Database/fed_database.json"

if os.path.exists(f'{system}/MS-Thesis Data/Database/Fed Data/keyword_info_ts.csv'):
    df_keyword_info_ts = pd.read_csv(f'{system}/MS-Thesis Data/Database/Fed Data/keyword_info_ts.csv')
else:
    df_keyword_info_ts = []
    
if os.path.exists(f'{system}/MS-Thesis Data/Database/Fed Data/keyword_freq_ts.json'):
    keyword_freq_ts = load_data(f"{system}/MS-Thesis Data/Database/Fed Data/keyword_freq_ts.json")
else:
    keyword_freq_ts = []
    
fed_funds = pd.read_excel(f'{system}/MS-Thesis Data/Database/Fed Data/FedFundsRate.xlsx', sheet_name='Monthly')
sp500 = pd.read_csv(f'{system}/MS-Thesis Data/Database/Market Data/GSPC.csv')

#Varaibles
