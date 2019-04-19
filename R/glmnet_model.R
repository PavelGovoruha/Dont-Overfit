library(tidyverse)
library(glmnet)
library(Matrix)

train <- read_rds('results/new_train.rds')
test <- read_rds('results/new_test.rds')
submission <-read_csv('data/sample_submission.csv')
x <- train[,3:ncol(train)]
y <- train$target

x_test <- test[,3:ncol(test)]

x_matrix <- Matrix(as.matrix(x), sparse = TRUE)
x_test_matrix <- Matrix(as.matrix(x_test), sparse = TRUE)

set.seed(2020)

glmnet_model_lasso <- cv.glmnet(x = x_matrix,
                          y = as.factor(y),
                          alpha = 1,
                          family = 'binomial',
                          type.measure = 'auc')
plot(glmnet_model_lasso)

glmnet_model_ridge <- cv.glmnet(x = x_matrix,
                                y = as.factor(y), 
                                alpha = 0,
                                family = 'binomial',
                                type.measure = 'auc')
plot(glmnet_model_ridge)

#Let's make predictions
pred_test_lasso <- predict(glmnet_model_lasso, x_test_matrix, type='response')
pred_test_ridge <- predict(glmnet_model_ridge, x_test_matrix,  type = 'response')

subm_lasso <- submission
subm_lasso$target <- pred_test_lasso

subm_ridge <- submission
subm_ridge$target <- pred_test_ridge

head(subm_lasso)
head(subm_ridge)

summary(subm_lasso$target)

names(subm_lasso)

write_csv(subm_lasso, 'results/submit_lasso.csv')
write_csv(subm_ridge, 'results/submit_ridge.csv')

