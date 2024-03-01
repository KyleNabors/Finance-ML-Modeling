library(stargazer)

setwd('/Users/kylenabors/Documents/Database')
sentiment = read.csv("/Users/kylenabors/Documents/Database/Models/FinBERT Models/taylor.csv")


reg0 <- lm(taylor ~ minute_sentiment_0 + sp500_return_0, data=sentiment)
reg1 <- lm(taylor ~ minute_sentiment_1 + sp500_return_1, data=sentiment)   
reg2 <- lm(taylor ~ minute_sentiment_2 + sp500_return_2, data=sentiment)
reg3 <- lm(taylor ~ minute_sentiment_3 + sp500_return_3, data=sentiment)
reg4 <- lm(taylor ~ minute_sentiment_4 + sp500_return_4, data=sentiment)


stargazer(reg0, reg1, reg2, reg3, reg4,
          column.labels = c("0 Months", "0-3 Months", "3-6 Months", "6-9 Months", "9-12 Months"),
          title ="Regressed Taylor On Michigan Sentiment (Column Lables Show X variable Lag)",
          #covariate.labels = c("Current", "0-3 Months Lagged", "3-6 Months Lagged", "6-9 Months Lagged", "9-12 Months Lagged"),
          out = "taylor.html")


reg0 <- lm(euro_funds ~ mpd_sentiment_0, data=sentiment)
reg1 <- lm(euro_funds ~ mpd_sentiment_1, data=sentiment)   
reg2 <- lm(euro_funds ~ mpd_sentiment_2, data=sentiment)
reg3 <- lm(euro_funds ~ mpd_sentiment_3, data=sentiment)
reg4 <- lm(euro_funds ~ mpd_sentiment_4, data=sentiment)


stargazer(reg0, reg1, reg2, reg3, reg4,
          column.labels = c("0 Months", "0-3 Months", "3-6 Months", "6-9 Months", "9-12 Months"),
          title ="Regressed Taylor On Michigan Sentiment (Column Lables Show X variable Lag)",
          #covariate.labels = c("Current", "0-3 Months Lagged", "3-6 Months Lagged", "6-9 Months Lagged", "9-12 Months Lagged"),
          out = "taylor_euro.html")


#covariate.labels = c("Current M", "Current S", "0-3 Months Lagged M", "0-3 Months Lagged S", "3-6 Months Lagged M", "3-6 Months Lagged S", "6-9 Months Lagged M", "6-9 Months Lagged S", "9-12 Months Lagged M", "9-12 Months Lagged S"),

#reg5 <- lm(taylor ~ minute_sentiment_5 + unemployment + inflation, data=sentiment)
#reg6 <- lm(taylor ~ minute_sentiment_6 + unemployment + inflation, data=sentiment)
#reg7 <- lm(taylor ~ minute_sentiment_7 + unemployment + inflation, data=sentiment)
#reg8 <- lm(taylor ~ minute_sentiment_8 + unemployment + inflation, data=sentiment)

#stargazer(reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8,
#          column.labels = c("0 Months", "0-3 Months", "3-6 Months", "6-9 Months", "9-12 Months", "12-15 Months", "15-18 Months", "18-21 Months","21-24 Months"),
#          out = "taylor.html")

reg0s <- lm(minute_sentiment_0 ~ sp500_return_0, data=sentiment)
reg1s <- lm(minute_sentiment_0 ~ sp500_return_1, data=sentiment)  
reg2s <- lm(minute_sentiment_0 ~ sp500_return_2, data=sentiment)
reg3s <- lm(minute_sentiment_0 ~ sp500_return_3, data=sentiment)
reg4s <- lm(minute_sentiment_0 ~ sp500_return_4, data=sentiment)
reg5s <- lm(statement_sentiment_0 ~ sp500_return_0, data=sentiment)
reg6s <- lm(statement_sentiment_0 ~ sp500_return_1, data=sentiment)   
reg7s <- lm(statement_sentiment_0 ~ sp500_return_2, data=sentiment)
reg8s <- lm(statement_sentiment_0 ~ sp500_return_3, data=sentiment)
reg9s <- lm(statement_sentiment_0 ~ sp500_return_4, data=sentiment)
stargazer(reg0s, reg1s, reg2s, reg3s, reg4s, reg5s, reg6s, reg7s, reg8s, reg9s,
          title ="Change in Minute and Statement Sentiment on SP500 Returns",
          covariate.labels = c("Current", "0-3 Months Lagged", "3-6 Months Lagged", "6-9 Months Lagged", "9-12 Months Lagged"),
          out = "sentiment.html")

regout <- lm(gdp ~ michigan_sentiment, data=sentiment)
summary(regout)