#Load libraries
library(tidyverse)
library(caret)
library(caretEnsemble)
library(future)
library(ggplot2)
library(gridExtra)

#Load train and test datasets with selected variables
train <- read_rds('data/train_new.rds')
test <- read_rds('data/test_new.rds')
submission <- read_csv('data/sample_submission.csv')

lsvm_ <- read_csv('results/caret_lsvm.csv')
glmnet_ <- read_csv('results/caret_glm.csv')

submission$target <- (lsvm_$target +glmnet_$target)/2

write_csv(submission, 'results/ensembling.csv')
