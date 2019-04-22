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

tune.glm <- expand.grid(lambda = seq(from = 0, to = 0.2, by = 0.01), alpha = c(0,1))

plan(multiprocess)
time1 <- Sys.time()
glmnet_model <- train(target ~ ., method = 'glmnet', metric = 'ROC', data = train[,-1], 
                      trControl = control.glm, tuneGrid = tune.glm)
Sys.time() - time1

plot(glmnet_model)

pred_test_raw <- predict(glmnet_model, test, type = 'raw')
pred_test_prob <- predict(glmnet_model, test, type = 'prob')
submission$target <- pred_test_prob$Y
write_csv(submission, 'data/caret_glm.csv')
#cforest model

#glmnet model
control.cf <- trainControl(method = 'boot', number = 100, savePredictions = 'final',
                            classProbs = TRUE, summaryFunction = twoClassSummary, 
                            returnResamp = 'final', allowParallel = TRUE)

tune.cf <- expand.grid(mtry = 8:10)

plan(multiprocess)
time1 <- Sys.time()
cforest_model <- train(target ~ ., method = 'cforest', metric = 'ROC', data = train[,-1], 
                      trControl = control.cf, tuneGrid = tune.cf)
Sys.time() - time1
plot(cforest_model)
cforest_model
pred_test_raw <- predict(cforest_model, test, type = 'raw')
pred_test_raw[1:10]
pred_test_prob <- predict(cforest_model, test, type = 'prob')
pred_test_prob[1:20,]
submission$target <- pred_test_prob$Y
write_csv(submission, 'results/caret_cforest.csv')


#glmnet boot ROC: 0.7574331, cforest boot ROC: 0.7134685