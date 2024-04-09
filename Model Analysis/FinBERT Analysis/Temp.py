import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np

# Define the file paths
meeting_minutes_path = (
    "/Users/kylenabors/Documents/Database/Training Data/fed/minutes/meeting_minutes.csv"
)
communications_path = "/Users/kylenabors/Downloads/communications.csv"

# Read the meeting minutes data
mins = pd.read_csv(meeting_minutes_path)
print(mins.columns)

# Read the communications data
mins_new = pd.read_csv(communications_path)
mins_new = mins_new[mins_new["Type"] == "Minute"]
mins_new = mins_new.rename(
    columns={"Date": "meeting_date", "Release Date": "release_date", "Text": "text"}
)
mins_new = mins_new[mins_new["release_date"] > "2023-07-05"]
print(mins_new.columns)

# combine the two dataframes and combine the text columns
minutes = pd.concat([mins, mins_new])
print(minutes.columns)

print(minutes["meeting_date"].min())
print(minutes["meeting_date"].max())

minutes.to_csv(
    "/Users/kylenabors/Documents/Database/Training Data/fed/minutes/meeting_minutes.csv"
)
