import os 
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np


fed_min = pd.read_csv('/Users/kylenabors/Documents/Database/Models/FinBERT Models/Fed/Minutes/Fed_Minutes_finbert_model_short.csv')
fed_min = fed_min[['date', 'sentiment']]
fed_min = fed_min.groupby('date').mean().reset_index()
fed_min = fed_min.rename(columns={'date': 'date', 'sentiment': 'Fed Minutes Sentiment'})


fed_bb = pd.read_csv('/Users/kylenabors/Documents/Database/Models/FinBERT Models/Fed/Beige Book/Fed_Beige Book_finbert_model_short.csv')
fed_bb = fed_bb[['date', 'sentiment']]
fed_bb = fed_bb.groupby('date').mean().reset_index()
fed_bb = fed_bb.rename(columns={'date': 'date', 'sentiment': 'Fed Beige Book Sentiment'})

fed_speeches = pd.read_csv('/Users/kylenabors/Documents/Database/Models/FinBERT Models/Fed/Speeches/Fed_Speeches_finbert_model_short.csv')
fed_speeches = fed_speeches[['date', 'sentiment']]
fed_speeches = fed_speeches.groupby('date').mean().reset_index()
fed_speeches = fed_speeches.rename(columns={'date': 'date', 'sentiment': 'Fed Speeches Sentiment'})

fed_state = pd.read_csv('/Users/kylenabors/Documents/Database/Models/FinBERT Models/Fed/Statements/Fed_Statements_finbert_model_short.csv')
fed_state = fed_state[['date', 'sentiment']]
fed_state = fed_state.groupby('date').mean().reset_index()
fed_state = fed_state.rename(columns={'date': 'date', 'sentiment': 'Fed Statements Sentiment'})

ecb_econ_bullitin = pd.read_csv('/Users/kylenabors/Documents/Database/Models/FinBERT Models/ECB/Economic Bulletin/ECB_Economic Bulletin_finbert_model_short.csv')
ecb_econ_bullitin = ecb_econ_bullitin[['date', 'sentiment']]
ecb_econ_bullitin = ecb_econ_bullitin.groupby('date').mean().reset_index()
ecb_econ_bullitin = ecb_econ_bullitin.rename(columns={'date': 'date', 'sentiment': 'ECB Economic Bulletin Sentiment'})

ecb_mon_policy_dec = pd.read_csv('/Users/kylenabors/Documents/Database/Models/FinBERT Models/ECB/Monetary policy decisions/ECB_Monetary policy decisions_finbert_model_short.csv')
ecb_mon_policy_dec = ecb_mon_policy_dec[['date', 'sentiment']]
ecb_mon_policy_dec = ecb_mon_policy_dec.groupby('date').mean().reset_index()
ecb_mon_policy_dec = ecb_mon_policy_dec.rename(columns={'date': 'date', 'sentiment': 'ECB Monetary Policy Decisions Sentiment'})

ecb_speeches = pd.read_csv('/Users/kylenabors/Documents/Database/Models/FinBERT Models/ECB/Speeches/ECB_Speeches_finbert_model_short.csv')
ecb_speeches = ecb_speeches[['date', 'sentiment']]
ecb_speeches = ecb_speeches.groupby('date').mean().reset_index()
ecb_speeches = ecb_speeches.rename(columns={'date': 'date', 'sentiment': 'ECB Speeches Sentiment'})


output = pd.merge(fed_min, fed_bb, how='outer', on='date')
output = pd.merge(output, fed_speeches, how='outer', on='date')
output = pd.merge(output, fed_state, how='outer', on='date')
output = pd.merge(output, ecb_econ_bullitin, how='outer', on='date')
output = pd.merge(output, ecb_mon_policy_dec, how='outer', on='date')
output = pd.merge(output, ecb_speeches, how='outer', on='date')

output = output.sort_values(by='date')

output.to_csv('/Users/kylenabors/Documents/Database/Models/FinBERT Models/All Sentiment Data.csv', index=False)