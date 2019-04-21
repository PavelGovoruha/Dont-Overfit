library(tidyverse)

lgb_ <- read_csv('results/lgb_model2.csv')
lasso_ <- read_csv('results/submit_lasso.csv')

lgb_lasso <- lgb_

lgb_lasso$target <- (lgb_$target + lasso_$target)/2

write_csv(lgb_lasso, 'results/lgb_lasso.csv')
