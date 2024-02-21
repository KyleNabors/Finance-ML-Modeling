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
          covariate.labels = c("Current Minute Sentiment", "Minute Sentiment 0-3 Months Ago", "Minute Sentiment 3-6 Months Ago", "Minute Sentiment 6-9 Months Ago", "Minute Sentiment 9-12 Months Ago", "Unemployment Rate", "Inflation Rate"),
          out = "taylor.html")

#reg5 <- lm(taylor ~ minute_sentiment_5 + unemployment + inflation, data=sentiment)
#reg6 <- lm(taylor ~ minute_sentiment_6 + unemployment + inflation, data=sentiment)
#reg7 <- lm(taylor ~ minute_sentiment_7 + unemployment + inflation, data=sentiment)
#reg8 <- lm(taylor ~ minute_sentiment_8 + unemployment + inflation, data=sentiment)

#stargazer(reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8,
#          column.labels = c("0 Months", "0-3 Months", "3-6 Months", "6-9 Months", "9-12 Months", "12-15 Months", "15-18 Months", "18-21 Months","21-24 Months"),
#          out = "taylor.html")