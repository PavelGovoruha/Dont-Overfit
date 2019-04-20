library(tidyverse)
library(lightgbm)


#Read data
train <- read_rds('results/train_new.rds')
test <- read_rds('results/test_new.rds')
submission <-read_csv('data/sample_submission.csv')

x_train <- train %>% select(-id, -target) %>% as.matrix()
y_train <- train$target

x_test <- test %>% select(-id) %>% as.matrix()

#Create lgb datasets
dtrain <- lgb.Dataset(data = x_train, label = y_train)

#Set parameters
params <- list(objective = "binary", 
               boost="gbdt",
               metric="auc",
               num_threads= 4,
               learning_rate = 0.01,
               num_leaves = 24,
               max_depth= 8,
               feature_fraction = 0.8,
               bagging_freq = 5,
               bagging_fraction = 0.8,
               min_data_in_leaf = 10,
               min_sum_hessian_in_leaf = 1e-3,
               lambda_l1 = 0.0001,
               lambda_l2 = 0.0001,
               verbosity = 1)
set.seed(1234)
lgb_cv <- lgb.cv(params,
                 data = dtrain,
                 nrounds=10000,
                 nfold = 10,
                 early_stopping_rounds = 100,
                 eval_freq=10,
                 seed=1234)

lgb_cv$best_score
lgb_cv$best_iter


