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

#bayesian glm
bayes_glm.control <- trainControl(method = 'boot', number = 100,savePredictions = 'final',
                                         classProbs = TRUE, summaryFunction = twoClassSummary, 
                                         returnResamp = 'final', allowParallel = TRUE)
plan(multiprocess)
time1 <- Sys.time()
bayes_glm_model <- train(target ~ ., method = 'bayesglm', metric = 'ROC', data = train_rf[,-1], 
                       trControl = bayes_glm.control)
Sys.time() - time1
bayes_glm_model

#bayesglm auc score: 0.84

#generalised additive models with splines
gener_spline.control <- trainControl(method = 'boot', number = 25,savePredictions = 'final',
                                     classProbs = TRUE, summaryFunction = twoClassSummary, 
                                     returnResamp = 'final', allowParallel = TRUE)

gener_spline.tune <- expand.grid(df = 1)
plan(multiprocess)
time1 <- Sys.time()
gener_spline_model <- train(target ~ ., method = 'gamSpline', metric = 'ROC', data = train_rf[,-1], 
                         trControl = gener_spline.control, tuneGrid = gener_spline.tune)
Sys.time() - time1
gener_spline_model

plot(gener_spline_model)

#generalised additive models with splines auc score: 0.823

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

#Next time: 
#1. Ensembles of Generalized Linear Models
#2. Boosted Generalized Linear Model
#3. Distance Weighted Discrimination with Radial Basis Function Kernel
#4. Penalized Discriminant Analysis
#5. ROC-Based Classifier
#6. Generalised additive models with splines

#Predict with baysglm
pred_test_bayes <- predict(bayes_glm_model, test_rf[,-1], type = 'raw')
pred_test_bayes[1:10]
pred_test_bayes_probs <- predict(bayes_glm_model, test_rf[,-1], type = 'prob')
pred_test_bayes_probs[1:10,]
submission$target <- pred_test_bayes_probs$Y
write_csv(submission, 'results/bayesglm_caret.csv')

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

#Predict with spline_model
pred_test_spline_raw <- predict(gener_spline_model, test_rf[,-1], type = 'raw')
pred_test_spline_raw[1:10]
pred_test_spline_prob <- predict(gener_spline_model, test_rf[,-1], type = 'prob')
pred_test_spline_prob[1:10,]
submission <- read_csv('data/sample_submission.csv')
submission$target <- pred_test_spline_prob$Y
write_csv(submission, 'results/spline_caret.csv')

#Basic ensembling
