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
tpt_month_change <- read_csv('/Users/kylenabors/Documents/Database/Models/BERT Models/tpt change monthly merged.csv')
summarise(tpt_month)
tot <- read_csv("/Users/kylenabors/Documents/Database/Models/BERT Models/pivot_df_tot.csv")
colnames(tot)[2] = "Topic_1"

tpt_month[is.na(tpt_month)] <- 0
tpt_month_change[is.na(tpt_month_change)] <- 0
#tpt_month_change[is.na(tpt_month_change)] <- 0
tpt_month <- as.data.frame(tpt_month)
tpt_month_change <- as.data.frame(tpt_month_change)
tpt_month <- slide(tpt_month, "FEDFUNDS", NewVar = "Lagged", slideBy = 1)  # create lead1 variable
tpt_month_change <- slide(tpt_month_change, "FEDFUNDS_Change", NewVar = "Lagged", slideBy = 1)  # create lead1 variable

plt_stock_value <- ggplot(tot, aes(x=Timestamp, y=Topic_1)) +
  geom_line()

plt_stock_value


reg_policy <- lm(Lagged ~ Inflation + Bank + Employment + Spending + Uncertainty, data = tpt_month)
summary(reg_policy)

reg_policy_change <- lm(Lagged ~ Inflation + Bank + Employment + Spending + Uncertainty, data = tpt_month_change)
summary(reg_policy_change)

reg1 <- lm(FEDFUNDS_Change ~ Employment + Housing + Banking + Inflation + Agriculture + Transportation + Growth + Oil + CPIENGSL, data = tpt_month_change)
summary(reg1)
reg2 <- lm(Lagged ~ Employment_Mean_Diff + Housing_Mean_Diff + Banking_Mean_Diff + Inflation_Mean_Diff + Agriculture_Mean_Diff + Transportation_Mean_Diff + Growth_Mean_Diff + Oil_Mean_Diff, data = tpt_month)
summary(reg2)




