library(stargazer)

setwd('/Users/kylenabors/Documents/Database')
sentiment = read.csv("/Users/kylenabors/Documents/Database/Models/FinBERT Models/taylor.csv")



reg0 <- lm(taylor ~ minute_sentiment_0 + unemployment + inflation, data=sentiment)
reg1 <- lm(taylor ~ minute_sentiment_1 + unemployment + inflation, data=sentiment)   
reg2 <- lm(taylor ~ minute_sentiment_2 + unemployment + inflation, data=sentiment)
reg3 <- lm(taylor ~ minute_sentiment_3 + unemployment + inflation, data=sentiment)
reg4 <- lm(taylor ~ minute_sentiment_4 + unemployment + inflation, data=sentiment)



stargazer(reg0, reg1, reg2, reg3, reg4,
          #column.labels = c("0 Months", "0-3 Months", "3-6 Months", "6-9 Months", "9-12 Months"),
          title ="Minute Sentment on Taylor Rule",
          covariate.labels = c("Current", "0-3 Months Lagged", "3-6 Months Lagged", "6-9 Months Lagged", "9-12 Months Lagged", "Unemployment Rate", "Inflation Rate"),
          out = "taylor.html")

#reg5 <- lm(taylor ~ minute_sentiment_5 + unemployment + inflation, data=sentiment)
#reg6 <- lm(taylor ~ minute_sentiment_6 + unemployment + inflation, data=sentiment)
#reg7 <- lm(taylor ~ minute_sentiment_7 + unemployment + inflation, data=sentiment)
#reg8 <- lm(taylor ~ minute_sentiment_8 + unemployment + inflation, data=sentiment)

#stargazer(reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8,
#          column.labels = c("0 Months", "0-3 Months", "3-6 Months", "6-9 Months", "9-12 Months", "12-15 Months", "15-18 Months", "18-21 Months","21-24 Months"),
#          out = "taylor.html")

reg0s <- lm(minute_sentiment ~    pce_0, data=sentiment)
reg1s <- lm(minute_sentiment ~    pce_1, data=sentiment)   
reg2s <- lm(minute_sentiment ~    pce_2, data=sentiment)
reg3s <- lm(minute_sentiment ~    pce_3, data=sentiment)
reg4s <- lm(minute_sentiment ~    pce_4, data=sentiment)
reg5s <- lm(statement_sentiment ~ pce_0, data=sentiment)
reg6s <- lm(statement_sentiment ~ pce_1, data=sentiment)   
reg7s <- lm(statement_sentiment ~ pce_2, data=sentiment)
reg8s <- lm(statement_sentiment ~ pce_3, data=sentiment)
reg9s <- lm(statement_sentiment ~ pce_4, data=sentiment)
stargazer(reg0s, reg1s, reg2s, reg3s, reg4s, reg5s, reg6s, reg7s, reg8s, reg9s,
          title ="PCE on Sentiment",
          covariate.labels = c("Current", "0-3 Months Lagged", "3-6 Months Lagged", "6-9 Months Lagged", "9-12 Months Lagged"),
          out = "sentiment.html")