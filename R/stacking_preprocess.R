#Load libraries
library(tidyverse)
library(caret)
library(future)
library(ggplot2)
library(gridExtra)

#Load data with selected variables
train <- read_rds('data/train_new.rds')
test <- read_rds('data/test_new.rds')

#Set random seed
set.seed(1234)

#Set train control param
control <- trainControl(method = 'boot', number = 25, savePredictions = 'final',
                       classProbs = TRUE, summaryFunction = twoClassSummary, 
                       returnResamp = 'final', allowParallel = TRUE)

#Create new train and test set which used in ensembling
train_meta <- train %>%
  mutate(glmnet_ = NA,
         lsvm_ = NA)
test_meta <- test %>%
  mutate(glmnet_ = NA,
         lsvm_ = NA)
#Check datasets
glimpse(train_meta)
glimpse(test_meta)

#Create folds
folds_indexes <- createFolds(y = train$target, k = 5)

#Print folds_indexes
folds_indexes

#Set tune parameters for glmnet and linear svm
tune_glm <- expand.grid(lambda = seq(from = 0, to = 1, by = 0.01), alpha = 0)
lsvm_tune <- expand.grid(cost = seq(0, 1, by = 0.05), weight = seq(0, 1, by = 0.1))

#Fill column glmnet_ in train meta
for(j in 1:5){
  temp_glmnet <- train(target ~ ., data = train[-folds_indexes[[j]], -1], 
                       method = 'glmnet', metric = 'ROC',
                       trControl = control, tuneGrid = tune_glm)
  pred <- predict(temp_glmnet, train[folds_indexes[[j]],-1], type = 'prob')
  train_meta$glmnet_[folds_indexes[[j]]] <- pred$Y
}

#Fill column lsvm_ in train meta
for(j in 1:5){
  temp_lsvm <- train(target ~ ., data = train[-folds_indexes[[j]], -1], 
                       method = 'svmLinearWeights', metric = 'ROC',
                       trControl = control, tuneGrid = lsvm_tune,
                       preProcess = 'center')
  pred <- predict(temp_lsvm, train[folds_indexes[[j]],-1], type = 'prob')
  train_meta$lsvm_[folds_indexes[[j]]] <- pred$Y
}

#Save meta train dataset
train_meta %>% write_rds('data/train_meta.rds')

#Train models on full train dataset and make predictions for test
glmnet_model <- train(target ~ ., method = 'glmnet', metric = 'ROC', data = train[,-1], 
                      trControl = control, tuneGrid = tune_glm)

pred_glm <- predict(glmnet_model, test[,-1], type = 'prob')
test_meta$glmnet_ <- pred_glm$Y

lsvm_model <- train(target ~ ., method = 'svmLinearWeights', metric = 'ROC', data = train[,-1],
                    trControl = control,
                    tuneGrid = lsvm_tune,
                    preProcess = 'center')

pred_lsvm <- predict(lsvm_model, test[,-1], type = 'prob')
test_meta$lsvm_ <- pred_lsvm$Y

#Save meta test dataset
test_meta %>% write_rds('data/test_meta.rds')

#Save folds indexes
folds_indexes %>% write_rds('data/folds_indexes.rds')
