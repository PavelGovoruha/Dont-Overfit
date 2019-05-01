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
         rsvm_ = NA)
test_meta <- test %>%
  mutate(pda_ = NA,
         rsvm_ = NA)
#Check datasets
glimpse(train_meta)
glimpse(test_meta)

#Create folds
folds_indexes <- createFolds(y = train$target, k = 5)

#Print folds_indexes
folds_indexes

#Set tune parameters for glmnet and linear svm
tune_pda <- expand.grid(lambda = seq(from = 0, to = 2, by = 0.01))
tune_svm <- expand.grid(C = seq(from = 0, to = 1, by = 0.25),
                        Weight = c(0, 0.36, 0.64, 1),
                        sigma = 0.04281106)

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
                       method = 'svmRadialWeights', metric = 'ROC',
                       trControl = control, tuneGrid = tune_svm,
                       preProcess = c('center', 'scale'))
  pred <- predict(temp_svm, train[folds_indexes[[j]],-1], type = 'prob')
  train_meta$rsvm_[folds_indexes[[j]]] <- pred$Y
}

#Check rsvm_ column
summary(train_meta$rsvm_)

#Save meta train dataset
train_meta %>% write_rds('data/train_meta.rds')

#Train models on full train dataset and make predictions for test
pda_model <- train(target ~ ., method = 'pda', metric = 'ROC', data = train[,-1], 
                      trControl = control, tuneGrid = tune_pda,
                   preProcess = 'scale')

pred_pda <- predict(pda_model, test[,-1], type = 'prob')
test_meta$pda_ <- pred_pda$Y

summary(test_meta$pda_)

svm_model <- train(target ~ ., method = 'svmRadialWeights', metric = 'ROC', 
                    data = train[,-1],
                    trControl = control,
                    tuneGrid = tune_svm,
                    preProcess = c('center', 'scale'))

pred_svm <- predict(svm_model, test[,-1], type = 'prob')

test_meta$rsvm_ <- pred_svm$Y

summary(test_meta$rsvm_)

#Save meta test dataset
test_meta %>% write_rds('data/test_meta.rds')
