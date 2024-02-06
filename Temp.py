import os 
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np


df = pd.read_csv("/Users/kylenabors/Documents/Database/Market Data/Unemployment/Unemployment.csv")
df = df.rename(columns = {'DATE':'date', 'UNRATE':'unemployment'})
print(df.head())


df.to_csv("/Users/kylenabors/Documents/Database/Market Data/Unemployment/Unemployment.csv", index = None, header=True)


# directory = '/Users/kylenabors/Downloads/Bloomberg Data-selected'

# # Check if the directory exists
# if not os.path.exists(directory):
#     print(f"The directory {directory} does not exist.")
#     sys.exit()

# for filename in os.listdir(directory):
#     if filename.endswith('.xlsx'): 
#         path = os.path.join(directory, filename)
#         print(path)
#         try:
#             df = pd.read_excel(path)
#             # Change the file extension to .csv
#             csv_filename = os.path.splitext(filename)[0] + '.csv'
#             csv_path = os.path.join(directory, csv_filename)
#             df.to_csv(csv_path, index = None, header=True)
#         except PermissionError:
#             print(f"Permission denied for file {path}.")
#         except Exception as e:
#             print(f"An error occurred while processing file {path}: {e}")