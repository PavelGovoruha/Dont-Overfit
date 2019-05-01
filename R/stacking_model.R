#Load libraries
library(tidyverse)
library(caret)
library(future)

#Load data
train <- read_rds('data/train_meta.rds')
test <- read_rds('data/test_meta.rds')
submission <- read_csv('data/sample_submission.csv')

#Check data
glimpse(train)
glimpse(test)

#Set random seed
set.seed(1234)

#Set grid
control <- trainControl(method = 'boot', number = 25,
                        savePredictions = 'final',
                        classProbs = TRUE, summaryFunction = twoClassSummary, 
                        returnResamp = 'final', allowParallel = TRUE)

#Train baysglm
plan(multiprocess)
time1 <- Sys.time()
model_bayesglm <- train(target ~ rsvm_ + pda_, method = 'bayesglm', 
                        data = train[,-1], metric = 'ROC', 
                        trControl = control)
Sys.time() - time1

model_bayesglm

#Predict with bayes glm
pred <- predict(model_bayesglm, test[,-1], type = 'prob')

submission$target <- pred$Y

submission %>% write_csv('results/bayes_stack.csv')
