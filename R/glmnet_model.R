library(tidyverse)
library(fastDummies)
library(glmnet)
library(Matrix)

train <- read_rds('results/new_train.rds')
test <- read_rds('results/new_test.rds')
submission <-read_csv('data/sample_submission.csv')
x <- train[,3:ncol(train)] %>% mutate(clust_4 = as.factor(clust_4))
y <- train$target

dummy_x <- dummy_columns(x) %>% mutate_if(is.integer, as.numeric) %>% select(-clust_4)

x_test <- test[,3:ncol(test)] %>% mutate(clust_4 = as.factor(clust_4))

dummy_test <- dummy_columns(x_test) %>% mutate_if(is.integer, as.numeric) %>% select(-clust_4)

x_matrix <- Matrix(as.matrix(dummy_x), sparse = TRUE)
x_test_matrix <- Matrix(as.matrix(dummy_test), sparse = TRUE)

set.seed(1234)

glmnet_model_lasso <- cv.glmnet(x = x_matrix,
                          y = as.factor(y),
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

