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
library("dplyr")

setwd('/Users/kylenabors/Documents/Database/Models/Sentiment Analysis Models/R')

sentiment = read_csv('/Users/kylenabors/Documents/Database/Models/Sentiment Analysis Models/Fed_Speeches_vs_ECB_Speeches_advanced_sentiment_texts.csv')

sentiment <- sentiment %>%                            
  dplyr::mutate(laggedval1 = lag(net_diff_tone, n = 1, default = NA)) 
  

sent_reg = lm(ECB_tone ~ Fed_tone, data = sentiment)
summary(sent_reg)
stargazer(sent_reg, out = 'Tone Regression.txt')

ts_model <- ts(sentiment$net_diff_tone)
ts_model
summary(ts_model)
#stargazer(ts_model, out = 'Time Series Regression.txt')

arima_model <- arima(sentiment$net_diff_tone, order=c(0,1,1))
arima_model
summary(arima_model)
stargazer(arima_model, out = 'ARIMA Regression.txt')

reg2 <- lm(net_diff_tone ~ laggedval1, data = sentiment)
summary(reg2)
stargazer(reg2, out = 'Tone Lagged Regression.txt')



plot_sentiment <- ggplot(sentiment, aes(x=ECB_compound, y=Fed_compound)) +
  geom_point()
plot_sentiment

plot_sentiment_time <- ggplot(sentiment, aes(x=monthly)) +
  geom_line(aes(y=ECB_compound)) + 
  geom_line(aes(y=Fed_compound)) 
plot_sentiment_time
