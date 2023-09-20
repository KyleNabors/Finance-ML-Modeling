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

setwd("/Users/kylenabors/Documents/GitHub/Finance-ML-Modeling")

tpt_month <- read_csv('/Users/kylenabors/Documents/Database/Models/BERT Models/tpt monthly merged.csv')
summarise(tpt_month)
tot <- read_csv("/Users/kylenabors/Documents/Database/Models/BERT Models/pivot_df_tot.csv")
colnames(tot)[2] = "Topic_1"

tpt_month[is.na(tpt_month)] <- 0
tpt_month <- as.data.frame(tpt_month)

tpt_month <- slide(tpt_month, "FEDFUNDS", NewVar = "Funds_1", slideBy = 1)  # create lead1 variable


plt_stock_value <- ggplot(tot, aes(x=Timestamp, y=Topic_1)) +
  geom_line()

plt_stock_value

reg1 <- lm(FEDFUNDS ~ Inflation + Bank + Employment + Spending + Uncertainty + CPIENGSL, data = tpt_month)
summary(reg1)