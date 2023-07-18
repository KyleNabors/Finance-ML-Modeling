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

sp500$interest <- ifelse(sp500$Keyword == 'interest', 1, 0)
sp500_interest <- sp500
sp500_interest = sp500_interest[sp500_interest$interest == 1, ]



reg_1 <- ts(sp500_interest)
reg_1

plot.ts(reg_1)