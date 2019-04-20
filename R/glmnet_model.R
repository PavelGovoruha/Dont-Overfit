library(tidyverse)
library(fastDummies)
library(glmnet)
library(Matrix)

train <- read_rds('results/train_new.rds')
test <- read_rds('results/test_new.rds')
submission <-read_csv('data/sample_submission.csv')

x_train <- train %>% select(-id, -target) 
y_train <- train$target

x_test <- test %>% select(-id)


x_train_matrix <- Matrix(as.matrix(x_train), sparse = TRUE)
x_test_matrix <- Matrix(as.matrix(x_test), sparse = TRUE)

set.seed(123456)

glmnet_model_lasso <- cv.glmnet(x = x_train_matrix,
                          y = as.factor(y_train),
                          alpha = 1,
                          family = 'binomial',
                          type.measure = 'auc')
plot(glmnet_model_lasso)

#Let's make predictions
pred_test_lasso <- predict(glmnet_model_lasso, x_test_matrix, type='response')

subm_lasso <- submission
subm_lasso$target <- pred_test_lasso

head(subm_lasso)

summary(subm_lasso$target)

write_csv(subm_lasso, 'results/submit_lasso.csv')

