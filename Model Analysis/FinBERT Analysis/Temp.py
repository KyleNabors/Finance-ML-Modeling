import os 
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np


mpd1 = pd.read_csv('/Users/kylenabors/Documents/Database/Training Data/ECB/Monetary policy decisions/Warin_Sanger_ECB.csv', encoding_errors='ignore', sep = ";")

mpd2 = pd.read_excel('/Users/kylenabors/Documents/Database/Training Data/ECB/Monetary policy decisions/Monetary policy decisions.xlsx')


mpd1 = mpd1[['date', 'firstPart']]
mpd1 = mpd1.rename(columns = {'firstPart': 'segment'})


mpd2 = mpd2[['date', 'title', 'segment']]

#convert date from yyyymmdd to yyyy-mm-dd
mpd2['date'] = mpd2['date'].astype(str)
mpd2['date'] = mpd2['date'].apply(lambda x: x[:4] + '-' + x[4:6] + '-' + x[6:])


print(mpd2.head())


mpd = pd.concat([mpd1, mpd2], axis = 0)

print(mpd.head())

mpd = mpd.sort_values(by = 'date')


mpd.to_csv('/Users/kylenabors/Documents/Database/Training Data/ECB/Monetary policy decisions/Monetary_policy_decisions.csv', index = False)