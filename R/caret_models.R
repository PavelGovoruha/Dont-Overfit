library(tidyverse)
library(caret)
library(future)
library(ggplot2)
library(gridExtra)

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

#glmnet model
tune_glm <- expand.grid(lambda = seq(from = 0, to = 1, by = 0.01), alpha = 0)

plan(multiprocess)
time1 <- Sys.time()
glmnet_model <- train(target ~ ., method = 'glmnet', metric = 'ROC', data = train[,-1], 
                      trControl = control, tuneGrid = tune_glm)
Sys.time() - time1

glmnet_model
plot(glmnet_model)
glmnet_model$bestTune
glmnet_model$results$ROC[6]

#glmnet local auc score: 0.8847

#Making prediction with glmnet model
pred_test_prob <- predict(glmnet_model, test[,-1], type = 'prob')

#Plot histogram and density of predicted probabilities
p1 <- qplot(pred_test_prob$Y, geom = 'density') + 
  ggtitle('Density of predictions')
p2 <- qplot(pred_test_prob$Y, geom = 'histogram') + ggtitle('Histogram of predictions')


p <- grid.arrange(p2, p1, ncol = 2)
ggsave(filename = 'plots/glmnet_predict.jpeg', plot = p, device = 'jpeg')

submission$target <- pred_test_prob$Y

write_csv(submission, 'results/caret_glm.csv')

#Linear SVM 
lsvm_tune <- expand.grid(cost = seq(0, 1, by = 0.05), weight = seq(0, 1, by = 0.1))

plan(multiprocess)
time1 <- Sys.time()
lsvm_model <- train(target ~ ., method = 'svmLinearWeights', metric = 'ROC', data = train[,-1],
                   trControl = control,
                   tuneGrid = lsvm_tune)
Sys.time() - time1
lsvm_model
plot(lsvm_model)
lsvm_model$bestTune
lsvm_model$results$ROC[160]
#Linear SVM local auc score : 0.8664202

#Predicting with Linear SVM
pred_test_prob <- predict(lsvm_model, test[,-1], type = 'prob')

#Plot histogram and density of predicted probabilities
p1 <- qplot(pred_test_prob$Y, geom = 'density') + 
  ggtitle('Density of predictions')
p2 <- qplot(pred_test_prob$Y, geom = 'histogram') + ggtitle('Histogram of predictions')

p <- grid.arrange(p2, p1, ncol = 2)
ggsave(filename = 'plots/lsvm_predict.jpeg', plot = p, device = 'jpeg')

submission$target <- pred_test_prob$Y

write_csv(submission, 'results/caret_lsvm.csv')