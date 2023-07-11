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
library(plotly)
library(leaflet)
library(hhi)
library(ggridges)
library(viridis)
library(hrbrthemes)



similar <- read_csv("/Users/kylenabors/Documents/GitHub/MS-Thesis/Models/Fed Models/One Model/similar_words.csv")

similar <- similar %>% arrange(Keyword)


# Convert the data to a long format
similar_long <- similar %>% 
  pivot_longer(cols = c("Similar Word", "Similarity"),
               names_to = "Variable",
               values_to = "Value")

# Print the reshaped DataFrame
print(similar_long)

