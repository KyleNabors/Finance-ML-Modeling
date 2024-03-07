library(stargazer)

setwd('/Users/kylenabors/Documents/Database')
sentiment = read.csv("/Users/kylenabors/Documents/Database/Models/FinBERT Models/taylor.csv")


reg0 <- lm(fedfunds ~ minute_sentiment_trend_0 + taylor_0, data=sentiment)
reg1 <- lm(fedfunds ~ minute_sentiment_trend_1 + taylor_1, data=sentiment)   
reg2 <- lm(fedfunds ~ minute_sentiment_trend_2 + taylor_2, data=sentiment)
reg3 <- lm(fedfunds ~ minute_sentiment_trend_3 + taylor_3, data=sentiment)
reg4 <- lm(fedfunds ~ minute_sentiment_trend_4 + taylor_4, data=sentiment)


stargazer(reg0, reg1, reg2, reg3, reg4,
          title ="Regressed ",
          #covariate.labels = c("Current", "1 Quarter Lagged", "2 Quarters Lagged", "3 Quarters Lagged", "4 Quarters Lagged"),
          out = "taylor.html")


reg0 <- lm(euro_funds ~ mpd_sentiment_trend_0 + taylor_euro_0, data=sentiment)
reg1 <- lm(euro_funds ~ mpd_sentiment_trend_1 + taylor_euro_1, data=sentiment)   
reg2 <- lm(euro_funds ~ mpd_sentiment_trend_2 + taylor_euro_2, data=sentiment)
reg3 <- lm(euro_funds ~ mpd_sentiment_trend_3 + taylor_euro_3, data=sentiment)
reg4 <- lm(euro_funds ~ mpd_sentiment_trend_4 + taylor_euro_4, data=sentiment)


stargazer(reg0, reg1, reg2, reg3, reg4,
          title ="Regressing Euro Taylor Rule on MPD Sentiment Trend",
          #covariate.labels = c("Current", "1 Quarter Lagged", "2 Quarters Lagged", "3 Quarters Lagged", "4 Quarters Lagged"),
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