import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np


ecb_speech1 = pd.read_csv(
    "/Users/kylenabors/Documents/Database/Training Data/ECB/Monetary policy decisions/Warin_Sanger_ECB.csv",
    encoding_errors="ignore",
    sep=";",
)

ecb_speech2 = pd.read_excel(
    "/Users/kylenabors/Documents/Database/Training Data/ECB/Monetary policy decisions/Monetary policy decisions.xlsx"
)


ecb_speech1 = ecb_speech1[["date", "firstPart"]]
ecb_speech1 = ecb_speech1.rename(columns={"firstPart": "segment"})


ecb_speech2 = ecb_speech2[["date", "title", "segment"]]

# convert date from yyyymmdd to yyyy-mm-dd
ecb_speech2["date"] = ecb_speech2["date"].astype(str)
ecb_speech2["date"] = ecb_speech2["date"].apply(
    lambda x: x[:4] + "-" + x[4:6] + "-" + x[6:]
)


print(ecb_speech2.head())


ecb_speech = pd.concat([ecb_speech1, ecb_speech2], axis=0)

print(ecb_speech.head())

ecb_speech = ecb_speech.sort_values(by="date")


ecb_speech.to_csv(
    "/Users/kylenabors/Documents/Database/Training Data/ECB/Monetary policy decisions/Monetary_policy_decisions.csv",
    index=False,
)


ecb_speech = ecb_speech.resample("Q", on="date").mean().reset_index()

filter_df = ecb_speech.copy(deep=True)
filter_df = filter_df[["date", "ecb_speech_sentiment"]]

cycle, trend = sm.tsa.filters.hpfilter(filter_df["ecb_speech_sentiment"], 1600)

filter_df["ecb_speech_sentiment_cycle"] = cycle
filter_df["ecb_speech_sentiment_trend"] = trend

filter_df = filter_df[["date", "ecb_speech_sentiment_cycle"]]
ecb_speech = ecb_speech.drop(columns=["ecb_speech_sentiment"])
filter_df = filter_df.rename(
    columns={"ecb_speech_sentiment_cycle": "ecb_speech_sentiment"}
)
ecb_speech = pd.merge(ecb_speech, filter_df, on="date", how="left")
ecb_speech = ecb_speech.groupby("date").mean().reset_index()
