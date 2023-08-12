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

setwd("/Users/kylenabors/Documents/GitHub/Finance-ML-Modeling")

keywords_freq <- read_csv("/Users/kylenabors/Documents/MS-Thesis Data/Database/Fed Data/keyword_info_ts.csv")
fed_funds <- read_csv("/Users/kylenabors/Documents/MS-Thesis Data/Database/Merged Data/merged_fed.csv")
sp500 <- read_csv("/Users/kylenabors/Documents/MS-Thesis Data/Database/Merged Data/merged_sp500.csv")
sp500_p1 <- read_csv("/Users/kylenabors/Documents/MS-Thesis Data/Database/Fed Data/Four Models/Merged Data/SP500/SP500 Merged Period 1.csv")
sp500_p2 <- read_csv("/Users/kylenabors/Documents/MS-Thesis Data/Database/Fed Data/Four Models/Merged Data/SP500/SP500 Merged Period 2.csv")
sp500_p3 <- read_csv("/Users/kylenabors/Documents/MS-Thesis Data/Database/Fed Data/Four Models/Merged Data/SP500/SP500 Merged Period 3.csv")
sp500_p4 <- read_csv("/Users/kylenabors/Documents/MS-Thesis Data/Database/Fed Data/Four Models/Merged Data/SP500/SP500 Merged Period 4.csv")

colnames(fed_funds)[1] = "Date"
colnames(sp500)[1] = "Date"



sp500$inflation <- ifelse(sp500$Keyword == 'inflation', 1, 0)
sp500$interest <- ifelse(sp500$Keyword == 'interest', 1, 0)
sp500$capital <- ifelse(sp500$Keyword == 'capital', 1, 0)
sp500$invest <- ifelse(sp500$Keyword == 'invest', 1, 0)
sp500$trade <- ifelse(sp500$Keyword == 'trade', 1, 0)
sp500$credit <- ifelse(sp500$Keyword == 'credit', 1, 0)
sp500$market <- ifelse(sp500$Keyword == 'market', 1, 0)

sp500_p1$inflation <- ifelse(sp500_p1$Keyword == 'inflation', 1, 0)
sp500_p1$interest <- ifelse(sp500_p1$Keyword == 'interest', 1, 0)
sp500_p1$capital <- ifelse(sp500_p1$Keyword == 'capital', 1, 0)
sp500_p1$invest <- ifelse(sp500_p1$Keyword == 'invest', 1, 0)
sp500_p1$trade <- ifelse(sp500_p1$Keyword == 'trade', 1, 0)
sp500_p1$credit <- ifelse(sp500_p1$Keyword == 'credit', 1, 0)
sp500_p1$market <- ifelse(sp500_p1$Keyword == 'market', 1, 0)

sp500_p2$inflation <- ifelse(sp500_p2$Keyword == 'inflation', 1, 0)
sp500_p2$interest <- ifelse(sp500_p2$Keyword == 'interest', 1, 0)
sp500_p2$capital <- ifelse(sp500_p2$Keyword == 'capital', 1, 0)
sp500_p2$invest <- ifelse(sp500_p2$Keyword == 'invest', 1, 0)
sp500_p2$trade <- ifelse(sp500_p2$Keyword == 'trade', 1, 0)
sp500_p2$credit <- ifelse(sp500_p2$Keyword == 'credit', 1, 0)
sp500_p2$market <- ifelse(sp500_p2$Keyword == 'market', 1, 0)

sp500_p3$inflation <- ifelse(sp500_p3$Keyword == 'inflation', 1, 0)
sp500_p3$interest <- ifelse(sp500_p3$Keyword == 'interest', 1, 0)
sp500_p3$capital <- ifelse(sp500_p3$Keyword == 'capital', 1, 0)
sp500_p3$invest <- ifelse(sp500_p3$Keyword == 'invest', 1, 0)
sp500_p3$trade <- ifelse(sp500_p3$Keyword == 'trade', 1, 0)
sp500_p3$credit <- ifelse(sp500_p3$Keyword == 'credit', 1, 0)
sp500_p3$market <- ifelse(sp500_p3$Keyword == 'market', 1, 0)

sp500_p4$inflation <- ifelse(sp500_p4$Keyword == 'inflation', 1, 0)
sp500_p4$interest <- ifelse(sp500_p4$Keyword == 'interest', 1, 0)
sp500_p4$capital <- ifelse(sp500_p4$Keyword == 'capital', 1, 0)
sp500_p4$invest <- ifelse(sp500_p4$Keyword == 'invest', 1, 0)
sp500_p4$trade <- ifelse(sp500_p4$Keyword == 'trade', 1, 0)
sp500_p4$credit <- ifelse(sp500_p4$Keyword == 'credit', 1, 0)
sp500_p4$market <- ifelse(sp500_p4$Keyword == 'market', 1, 0)

lm_sp500_value <- lm(Value ~ I(Frequency*inflation) + I(Frequency*interest) + I(Frequency*capital) + I(Frequency*invest) + I(Frequency*trade) + I(Frequency*credit) + I(Frequency*market), data = sp500)
summary(lm_sp500_value)

lm_sp500_change <- lm(Change ~ I(Frequency*inflation) + I(Frequency*interest) + I(Frequency*capital) + I(Frequency*invest) + I(Frequency*trade) + I(Frequency*credit) + I(Frequency*market), data = sp500)
summary(lm_sp500_change)

stargazer(lm_sp500_value, lm_sp500_change,
          title = "Linear Regression Model for Keyword Frequency Impact on S&P500 December 2008 - July 2023",
          dep.var.labels=c("S&P Value","S&P Day Change"),
          type="html",
          out = "Tables/One Model Linear Regression Table.htm"
          )

#sp500$Frequency <- log(sp500$Frequency)

lml_sp500_value <- lm(Value ~ I(Frequency*inflation) + I(Frequency*interest) + I(Frequency*capital) + I(Frequency*invest) + I(Frequency*trade) + I(Frequency*credit) + I(Frequency*market), data = sp500)
summary(lml_sp500_value)

lml_sp500_change <- lm(Change ~ I(Frequency*inflation) + I(Frequency*interest) + I(Frequency*capital) + I(Frequency*invest) + I(Frequency*trade) + I(Frequency*credit) + I(Frequency*market), data = sp500)
summary(lml_sp500_change)


lm_sp500_value_p1 <- lm(Value ~ I(Frequency*inflation) + I(Frequency*interest) + I(Frequency*capital) + I(Frequency*invest) + I(Frequency*trade) + I(Frequency*credit) + I(Frequency*market), data = sp500_p1)
summary(lm_sp500_value_p1)

lm_sp500_change_p1 <- lm(Change ~ I(Frequency*inflation) + I(Frequency*interest) + I(Frequency*capital) + I(Frequency*invest) + I(Frequency*trade) + I(Frequency*credit) + I(Frequency*market), data = sp500_p1)
summary(lm_sp500_change_p1)


lm_sp500_value_p2 <- lm(Value ~ I(Frequency*inflation) + I(Frequency*interest) + I(Frequency*capital) + I(Frequency*invest) + I(Frequency*trade) + I(Frequency*credit) + I(Frequency*market), data = sp500_p2)
summary(lm_sp500_value_p2)

lm_sp500_change_p2 <- lm(Change ~ I(Frequency*inflation) + I(Frequency*interest) + I(Frequency*capital) + I(Frequency*invest) + I(Frequency*trade) + I(Frequency*credit) + I(Frequency*market), data = sp500_p2)
summary(lm_sp500_change_p2)


lm_sp500_value_p3 <- lm(Value ~ I(Frequency*inflation) + I(Frequency*interest) + I(Frequency*capital) + I(Frequency*invest) + I(Frequency*trade) + I(Frequency*credit) + I(Frequency*market), data = sp500_p3)
summary(lm_sp500_value_p3)

lm_sp500_change_p3 <- lm(Change ~ I(Frequency*inflation) + I(Frequency*interest) + I(Frequency*capital) + I(Frequency*invest) + I(Frequency*trade) + I(Frequency*credit) + I(Frequency*market), data = sp500_p3)
summary(lm_sp500_change_p3)


lm_sp500_value_p4 <- lm(Value ~ I(Frequency*inflation) + I(Frequency*interest) + I(Frequency*capital) + I(Frequency*invest) + I(Frequency*trade) + I(Frequency*credit) + I(Frequency*market), data = sp500_p4)
summary(lm_sp500_value_p4)

lm_sp500_change_p4 <- lm(Change ~ I(Frequency*inflation) + I(Frequency*interest) + I(Frequency*capital) + I(Frequency*invest) + I(Frequency*trade) + I(Frequency*credit) + I(Frequency*market), data = sp500_p4)
summary(lm_sp500_change_p4)

regs <- list(lm_sp500_value_p1, lm_sp500_value_p2, lm_sp500_value_p3, lm_sp500_value_p4, lm_sp500_change_p1, lm_sp500_change_p2, lm_sp500_change_p3, lm_sp500_change_p4)

stargazer(regs,
          title = "Linear Regression Model for Keyword Frequency Impact on S&P500 Four Periods",
          dep.var.labels=c("S&P Value","S&P Day Change"),
          column.labels = c("Period 1", "Period 2", "Period 3", "Period 4", "Period 1", "Period 2", "Period 3", "Period 4"),
          model.numbers=FALSE,
          type="html",
          out = "Tables/Four Models Linear Regression Table.htm"
)


plt_keywords <- ggplot(sp500, aes(x=Date, y=Frequency, group=Keyword, color=Keyword)) +
  geom_line()

plt_stock_value <- ggplot(sp500, aes(x=Date, y=Value)) +
  geom_line()

plt_stock_change <- ggplot(sp500, aes(x=Date, y=Change)) +
  geom_line()


scalar <- 2.5
multiplot <- ggplot(sp500, aes(x=Date)) +
  geom_line( aes(y=Frequency, color=Keyword),linewidth = 1) +
  geom_line( aes(y=Value / scalar), linewidth = .3) +
  facet_wrap(~Keyword,
             scales = "fixed") +
  scale_y_continuous(name="Frequency of Keyword",
                     sec.axis = sec_axis(~.*scalar, name="Value of S&P500"))

multiplot

scalar2 <- .1
multiplot2 <- ggplot(sp500, aes(x=Date)) +
  geom_line( aes(y=Frequency, color=Keyword),linewidth = 1) +
  geom_line( aes(y=Change / scalar2), linewidth = .3) +
  facet_wrap(~Keyword,
             scales = "fixed") +
  scale_y_continuous(name="Frequency of Keyword",
                     sec.axis = sec_axis(~.*scalar2, name="Change of S&P500"))

multiplot2

sp500$inflationF <- sp500$Frequency*sp500$inflation
sp500$interestF <- sp500$Frequency*sp500$interest
sp500$capitalF <- sp500$Frequency*sp500$capital
sp500$investF <- sp500$Frequency*sp500$invest
sp500$tradeF <- sp500$Frequency*sp500$trade
sp500$creditF <- sp500$Frequency*sp500$credit
sp500$marketF <- sp500$Frequency*sp500$market

corr_matrix <- cor(sp500[,c("inflationF", "interestF", "capitalF", "investF", "tradeF", "creditF", "marketF")])
corr_matrix

stargazer(corr_matrix, type="html", out="Tables/Correlation Table.htm")

