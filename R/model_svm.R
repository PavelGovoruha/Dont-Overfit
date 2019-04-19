library(tidyverse)
library(kernlab)
library(fastDummies)

train <- read_rds('results/new_train.rds')
test <- read_rds('results/new_test.rds')
submission <-read_csv('data/sample_submission.csv')

