library(tidyverse)
library(caret)
library(future)

#Load train and test datasets with selected variables
train <- read_rds('data/train_new.rds')
test <- read_rds('data/test_new.rds')
submission <- read_csv('data/sample_submission.csv')

#Set random seed to repeat result
set.seed(1234)

#Set control parameters for caret training process
control <- trainControl(method = 'boot', number = 25, savePredictions = 'final',
                        classProbs = TRUE, summaryFunction = twoClassSummary, 
                        returnResamp = 'final', allowParallel = TRUE)

#Define tune grid for pda
tune_pda <- expand.grid(lambda = seq(from = 0, to = 2, by = 0.01))

#Train pda model
plan(multiprocess)
time1 <- Sys.time()
pda_model <- train(target ~ ., data = train[,-1], method = 'pda',
                   metric = 'ROC',
                   trControl = control,
                   tuneGrid = tune_pda,
                   preProcess = 'scale')
Sys.time() - time1

plot(pda_model)

#Best parameters
pda_model$bestTune

#Best local auc
max(pda_model$results$ROC)

#Make prediction with pda
pred_prob_pda <- predict(pda_model, test[,-1], type = 'prob')
submission$target <- pred_prob_pda$Y

submission %>% write_csv('results/caret_pda.csv')

#Define tune grid for radial svm with weights
tune_svm <- expand.grid(cost = seq(from = 0, to = 1, by = 0.01),
                                   weight = 0.12)
#train svm model
plan(multiprocess)
time1 <- Sys.time()
svm_model <- train(target ~ ., data = train[,-1],
                   method = 'svmLinearWeights',
                   metric = 'ROC',
                   trControl = control,
                   tuneGrid = tune_svm,
                   preProcess = c('center', 'scale'))
Sys.time() - time1

#Plot svm model
plot(svm_model)

#Best parameters
svm_model$bestTune

#Best local auc
max(svm_model$results$ROC)

#Make predictions with svm
pred_prob_svm <- predict(svm_model, test[,-1], type = 'prob')
submission$target <- pred_prob_svm$Y

submission %>% write_csv('results/caret_lsvm.csv')
