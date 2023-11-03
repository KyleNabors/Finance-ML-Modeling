library(readr)  
library(plyr)
library(nlme)
library(stringr)
library(readxl)
library(writexl)
library(stargazer)
library(tidyverse)
library(plm)
library(tseries)
library(zoo)
library(dplyr)
library(tidyr)
library(ggplot2)
library(DataCombine)
library(xts)

#setwd('/Users/kylenabors/Documents/Database/Models/Sentiment Analysis Models')

sentiment = read_csv('/Users/kylenabors/Documents/Database/Models/Sentiment Analysis Models/Fed_Speeches_vs_ECB_Speeches_advanced_sentiment_texts.csv')

sent_reg = lm(ECB_compound ~ Fed_compound, data = sentiment)
summary(sent_reg)
stargazer(sent_reg, out = 'Sentiment Regression.txt')

plot_sentiment <- ggplot(sentiment, aes(x=ECB_compound, y=Fed_compound)) +
  geom_point()
plot_sentiment

plot_sentiment_time <- ggplot(sentiment, aes(x=monthly)) +
  geom_line(aes(y=ECB_compound)) + 
  geom_line(aes(y=Fed_compound)) 
plot_sentiment_time
