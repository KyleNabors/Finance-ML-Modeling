import os 
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np



market = pd.merge(market, sp500, on='date', how='outer')
market = pd.merge(market, emini, on='date', how='outer')
market = pd.merge(market, eurodollar, on='date', how='outer')
market = pd.merge(market, fedfutures, on='date', how='outer')
market = pd.merge(market, vix, on='date', how='outer')
market = pd.merge(market, unemployment, on='date', how='outer')
market = pd.merge(market, inflation, on='date', how='outer')