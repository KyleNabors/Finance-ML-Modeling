rm(list=ls())


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
library(forecast)
library(AER)
library(dynlm)
library(scales)
library(quantmod)
library(urca)

setwd("/Users/kylenabors/Documents/GitHub/MS-Thesis")

keywords_freq <- read_csv("/Users/kylenabors/Documents/MS-Thesis Data/Database/Fed Data/keyword_info_ts.csv")
fed_funds <- read_csv("/Users/kylenabors/Documents/MS-Thesis Data/Database/Merged Data/merged_fed.csv")
sp500 <- read_csv("/Users/kylenabors/Documents/MS-Thesis Data/Database/Merged Data/merged_sp500.csv")

colnames(fed_funds)[1] = "Date"
colnames(sp500)[1] = "Date"
colnames(sp500)[5] = "value"
colnames(sp500)[6] = "change"

sp500$inflation <- ifelse(sp500$Keyword == 'inflation', 1, 0)
sp500$interest <- ifelse(sp500$Keyword == 'interest', 1, 0)
sp500$uncertain <- ifelse(sp500$Keyword == 'uncertain', 1, 0)
sp500$invest <- ifelse(sp500$Keyword == 'invest', 1, 0)
sp500$trade <- ifelse(sp500$Keyword == 'trade', 1, 0)

lm_sp500 <- lm(value ~ I(Frequency*inflation) + I(Frequency*interest) + I(Frequency*uncertain) + I(Frequency*invest) + I(Frequency*trade), data = sp500)
summary(lm_sp500)

sp500_interest <- sp500
sp500_interest = sp500_interest[sp500_interest$interest == 1, ]

sp_interest<- lm(change ~ Frequency, data = sp500_interest)
summary(sp_interest)

fed_funds$interest <- ifelse(fed_funds$Keyword == 'interest', 1, 0)
fed_funds_interest <- fed_funds
fed_funds_interest = fed_funds_interest[fed_funds_interest$interest == 1, ]

ff_interest <- lm(FEDFUNDS ~ Frequency, data = fed_funds_interest)
summary(ff_interest)

test <- lm(FEDFUNDS ~ I(interest*Frequency), data = fed_funds)
summary(test)

fed_funds$inflation <- ifelse(fed_funds$Keyword == 'inflation', 1, 0)
fed_funds_inflation <- fed_funds
fed_funds_inflation = fed_funds_inflation[fed_funds_inflation$inflation == 1, ]

fed_funds_ii <- merge(fed_funds_inflation, fed_funds_interest, by="Date")

ff_ii <- lm(FEDFUNDS.x ~ Frequency.y + Frequency.x, data = fed_funds_ii)
summary(ff_ii)
