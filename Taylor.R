

library(stargazer)

setwd('/Users/kylenabors/Documents/Database')
state = read.csv("/Users/kylenabors/Documents/Database/Models/FinBERT Models/taylor.csv")

summary(state)

reg <- lm(sentiment ~ sentiment_1 + taylor, data = state)
reg1 <- lm(sentiment_1 ~ sentiment_2 + taylor, data = state)
reg2 <- lm(sentiment_2 ~ sentiment_3 + taylor, data = state)
reg3 <- lm(sentiment_3 ~ sentiment_4 + taylor, data = state)
reg4 <- lm(sentiment_4 ~ sentiment_5 + taylor, data = state)
reg5 <- lm(sentiment_5 ~ sentiment_6 + taylor, data = state)
reg6 <- lm(sentiment_6 ~ sentiment_7 + taylor, data = state)
reg7 <- lm(sentiment_7 ~ sentiment_8 + taylor, data = state)
reg8 <- lm(sentiment_8 ~ sentiment_9 + taylor, data = state)

stargazer(reg, reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8,
          column.labels = c("0 Months", "0-3 Months", "3-6 Months", "6-9 Months", "9-12 Months", "12-15 Months", "15-18 Months", "21-24 Months", "24-27 Months"),
          out = "taylor.html")
