#Load libraries
library(tidyverse)
library(caret)
library(future)
library(ggplot2)
library(gridExtra)

#Load data
train <- read_rds('data/train_meta.rds')
test <- read_rds('data/test_meta.rds')
submission <- read_csv('data/sample_submission.csv')

#Check data
glimpse(train)
glimpse(test)

#Set caret control parameters
control <- trainControl(method = 'boot', number = 25, savePredictions = 'final',
                                   classProbs = TRUE, summaryFunction = twoClassSummary, 
                                   returnResamp = 'final', allowParallel = TRUE)

bayes_glm <- train(target ~ ., method = 'bayesglm', data = train[,-1],
                   metric = 'ROC',
                   trControl = control)
bayes_glm

svmRadial <- train(target ~ ., method = 'svmRadialWeights', metric = 'ROC',
                   data = train[,-1], trControl = control,
                   tuneLength = 10, preProcess = 'center')
svmRadial

#Predict with bayes glm
pred <- predict(bayes_glm, test[,-1], type = 'prob')
prob_bayes_glm <- pred$Y
summary(prob_bayes_glm)
submission$target <- prob_bayes_glm

submission %>% write_csv('results/stack_bayes.csv')

#Predict with Radial SVM
pred <- predict(svmRadial, test[,-1], type = 'prob')
prob_rsvm <- pred$Y
summary(prob_rsvm)
submission$target <- prob_rsvm

submission %>% write_csv('results/stack_rsvm.csv')
