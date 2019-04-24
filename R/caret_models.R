library(tidyverse)
library(caret)
library(future)

train_rf <- read_rds('data/train_sel_rf.rds')
test_rf <- read_rds('data/test_sel_rf.rds')
submission <- read_csv('data/sample_submission.csv')
set.seed(1234)

#glmnet model
control.glm <- trainControl(method = 'boot', number = 100,savePredictions = 'final',
                            classProbs = TRUE, summaryFunction = twoClassSummary, 
                            returnResamp = 'final', allowParallel = TRUE)

tune.glm <- expand.grid(lambda = seq(from = 0, to = 2, by = 0.01), alpha = 0)

plan(multiprocess)
time1 <- Sys.time()
glmnet_model <- train(target ~ ., method = 'glmnet', metric = 'ROC', data = train_rf[,-1], 
                      trControl = control.glm, tuneGrid = tune.glm)
Sys.time() - time1

plot(glmnet_model)

pred_test_raw <- predict(glmnet_model, test, type = 'raw')
pred_test_prob <- predict(glmnet_model, test, type = 'prob')
submission$target <- pred_test_prob$Y
write_csv(submission, 'data/caret_glm.csv')

#glmnet auc score: 0.8426

#Radial SVM
kern.control <- trainControl(method = 'boot', number = 100,savePredictions = 'final',
                             classProbs = TRUE, summaryFunction = twoClassSummary, 
                             returnResamp = 'final', allowParallel = TRUE)
kern.tune <- expand.grid(sigma = 10^(-5:-1), C = 10^(-3:3), Weight = c("Y" = 0.36, "N" = 0.64))
plan(multiprocess)
time1 <- Sys.time()
kern_model <- train(target ~ ., method = 'svmRadialWeights', metric = 'ROC', data = train_rf[,-1],
                    trControl = kern.control, tuneGrid = kern.tune)
Sys.time() - time1
kern_model
plot(kern_model)

#svmRadial auc score: 0.833

# Cforest 
cforest.control <- trainControl(method = 'boot', number = 100, savePredictions = 'final',
                             classProbs = TRUE, summaryFunction = twoClassSummary, 
                             returnResamp = 'final', allowParallel = TRUE)
cforest.tune <- expand.grid(mtry = 1:5)
plan(multiprocess)
time1 <- Sys.time()
cforest_model <- train(target ~ ., method = 'cforest', metric = 'ROC', data = train_rf[,-1],
                    trControl = cforest.control, tuneGrid = cforest.tune)
Sys.time() - time1
cforest_model
plot(cforest_model)

#Cforest model auc score: 0.822

#Predict by glmnet
pred_test_raw_glmnet <- predict(glmnet_model, test_rf[,-1], type = 'raw')
pred_test_raw_glmnet[1:10]
pred_test_prob_glmnet <- predict(glmnet_model, test_rf[,-1], type = 'prob')
pred_test_prob_glmnet[1:10,]
submission$target <- pred_test_prob_glmnet$Y
write_csv(submission, 'results/caret_glmnet.csv')

#Predict with kern_model
pred_test_kern_raw <- predict(kern_model, test_rf[,-1], type = 'raw')
pred_test_kern_raw[1:10]
pred_test_kern_prob <- predict(kern_model, test_rf[,-1], type = 'prob')
pred_test_kern_prob[1:10,]
submission <- read_csv('data/sample_submission.csv')
submission$target <- pred_test_kern_prob$Y
write_csv(submission, 'results/svm_caret.csv')

#Predict with cforest
pred_test_cforest_raw <- predict(cforest_model, test_rf[,-1], type = 'raw')
pred_test_cforest_raw[1:10]
pred_test_cforest_prob <- predict(cforest_model, test_rf[,-1], type = 'prob')
pred_test_cforest_prob[1:10,]
submission <- read_csv('data/sample_submission.csv')
submission$target <- pred_test_cforest_prob$Y
write_csv(submission, 'results/caret_cforest.csv')
