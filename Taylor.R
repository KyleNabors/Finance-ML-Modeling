library(stargazer)

setwd('/Users/kylenabors/Documents/Database')
sentiment = read.csv("/Users/kylenabors/Documents/Database/Models/FinBERT Models/taylor.csv")


sentiment$sp500_price = log10(sentiment$sp500_price)


#reg0 <- lm(taylor ~ fed_sentiment_0, data=sentiment)
#reg1 <- lm(taylor ~ fed_sentiment_1, data=sentiment)
#reg2 <- lm(taylor ~ fed_sentiment_2, data=sentiment)
#reg3 <- lm(taylor ~ fed_sentiment_3, data=sentiment)
#reg4 <- lm(taylor ~ fed_sentiment_4, data=sentiment)
#reg5 <- lm(taylor ~ fed_sentiment_5, data=sentiment)
#reg6 <- lm(taylor ~ fed_sentiment_6, data=sentiment)
#reg7 <- lm(taylor ~ fed_sentiment_7, data=sentiment)
#reg8 <- lm(taylor ~ fed_sentiment_8, data=sentiment)
#reg9 <- lm(taylor ~ fed_sentiment_9, data=sentiment)


reg0 <- lm(statement_sentiment_forward_0 ~ taylor, data=sentiment)
reg1 <- lm(statement_sentiment_forward_1 ~ taylor, data=sentiment)
reg2 <- lm(statement_sentiment_forward_2 ~ taylor, data=sentiment)
reg3 <- lm(statement_sentiment_forward_3 ~ taylor, data=sentiment)
reg4 <- lm(statement_sentiment_forward_4 ~ taylor, data=sentiment)
reg5 <- lm(statement_sentiment_forward_5 ~ taylor, data=sentiment)
reg6 <- lm(statement_sentiment_forward_6 ~ taylor, data=sentiment)
reg7 <- lm(statement_sentiment_forward_7 ~ taylor, data=sentiment)
reg8 <- lm(statement_sentiment_forward_8 ~ taylor, data=sentiment)
reg9 <- lm(statement_sentiment_forward_9 ~ taylor, data=sentiment)

stargazer(reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8, reg9,
          column.labels = c("0 Months", "0-3 Months", "3-6 Months", "6-9 Months", "9-12 Months", "12-15 Months", "15-18 Months", "18-21 Months","21-24 Months", "24-27 Months"),
          out = "taylor.html")




#reg0 <- lm(taylor ~ statement_sentiment_0 + statement_sentiment_1, data=sentiment)
#reg1 <- lm(taylor ~ statement_sentiment_1 + statement_sentiment_2, data=sentiment)
#reg2 <- lm(taylor ~ statement_sentiment_2 + statement_sentiment_3, data=sentiment)
#reg3 <- lm(taylor ~ statement_sentiment_3 + statement_sentiment_4, data=sentiment)
#reg4 <- lm(taylor ~ statement_sentiment_4 + statement_sentiment_5, data=sentiment)
#reg5 <- lm(taylor ~ statement_sentiment_5 + statement_sentiment_6, data=sentiment)
#reg6 <- lm(taylor ~ statement_sentiment_6 + statement_sentiment_7, data=sentiment)
#reg7 <- lm(taylor ~ statement_sentiment_7 + statement_sentiment_8, data=sentiment)
#reg8 <- lm(taylor ~ statement_sentiment_8 + statement_sentiment_9, data=sentiment)
#reg9 <- lm(taylor ~ statement_sentiment_9 + statement_sentiment_10, data=sentiment)


