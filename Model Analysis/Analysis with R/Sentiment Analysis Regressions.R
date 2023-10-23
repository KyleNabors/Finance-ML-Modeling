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

setwd('/Users/kylenabors/Documents/Database/Models/Sentiment Analysis Models')

sentiment = read_csv('ECB_Speeches_vs_Fed_Speeches_advanced_sentiment_texts.csv')

sent_reg = lm(net_1 ~ net_2, data = sentiment)
summary(sent_reg)
stargazer(sent_reg, out = 'Sentiment Regression.txt')

plot_sentiment <- ggplot(sentiment, aes(x=net_1, y=net_2)) +
  geom_point()
plot_sentiment

plot_sentiment_time <- ggplot(sentiment, aes(x=monthly)) +
  geom_line(aes(y=net_1)) + 
  geom_line(aes(y=net_2)) 
plot_sentiment_time
