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

setwd("/Users/kylenabors/Documents/GitHub/Finance-ML-Modeling")

tot <- read_csv("/Users/kylenabors/Documents/Database/Models/BERT Models/pivot_df_tot.csv")
colnames(tot)[2] = "Topic_1"

plt_stock_value <- ggplot(tot, aes(x=Timestamp, y=Topic_1)) +
  geom_line()

plt_stock_value