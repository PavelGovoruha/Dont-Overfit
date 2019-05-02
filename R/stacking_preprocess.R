#Load libraries
library(tidyverse)
library(caret)

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
  mutate(pda_ = NA,
         lsvm_ = NA)
test_meta <- test %>%
  mutate(pda_ = NA,
         lsvm_ = NA)
#Check datasets
glimpse(train_meta)
glimpse(test_meta)

#Create folds
folds_indexes <- createFolds(y = train$target, k = 5)

#Print folds_indexes
folds_indexes

#Set tune parameters for glmnet and linear svm
tune_pda <- expand.grid(lambda = 0.99)
tune_svm <- expand.grid(cost = 0.03,
                        weight = 0.12)

#Fill column pda_ in train meta
for(j in 1:5){
  temp_pda <- train(target ~ ., data = train[-folds_indexes[[j]], -1], 
                       method = 'pda', metric = 'ROC',
                       trControl = control, tuneGrid = tune_pda,
                       preProcess = 'scale')
  pred <- predict(temp_pda, train[folds_indexes[[j]],-1], type = 'prob')
  train_meta$pda_[folds_indexes[[j]]] <- pred$Y
}

#Check pda_ column
summary(train_meta$pda_)

#Fill column lsvm_ in train meta
for(j in 1:5){
  temp_svm <- train(target ~ ., data = train[-folds_indexes[[j]], -1], 
                       method = 'svmLinearWeights', metric = 'ROC',
                       trControl = control, tuneGrid = tune_svm,
                       preProcess = c('center', 'scale'))
  pred <- predict(temp_svm, train[folds_indexes[[j]],-1], type = 'prob')
  train_meta$lsvm_[folds_indexes[[j]]] <- pred$Y
}

#Check lsvm_ column
summary(train_meta$lsvm_)

#Save meta train dataset
train_meta %>% write_rds('data/train_meta.rds')

#Train models on full train dataset and make predictions for test
pda_model <- train(target ~ ., method = 'pda', metric = 'ROC', data = train[,-1], 
                      trControl = control, tuneGrid = tune_pda,
                   preProcess = 'scale')

pred_pda <- predict(pda_model, test[,-1], type = 'prob')
test_meta$pda_ <- pred_pda$Y

summary(test_meta$pda_)

svm_model <- train(target ~ ., method = 'svmLinearWeights', metric = 'ROC', 
                    data = train[,-1],
                    trControl = control,
                    tuneGrid = tune_svm,
                    preProcess = c('center', 'scale'))

pred_svm <- predict(svm_model, test[,-1], type = 'prob')

test_meta$lsvm_ <- pred_svm$Y

summary(test_meta$lsvm_)

#Save meta test dataset
test_meta %>% write_rds('data/test_meta.rds')

#Save folds
folds_indexes %>% write_rds('data/folds_indexes.rds')
