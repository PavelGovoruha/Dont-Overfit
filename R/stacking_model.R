#Load libraries
library(tidyverse)
library(caret)
library(future)
library(foreach)
library(MLmetrics)

#Load data
train <- read_rds('data/train_meta.rds')
test <- read_rds('data/test_meta.rds')
submission <- read_csv('data/sample_submission.csv')

#Check data
glimpse(train)
glimpse(test)

#Load folds indexes
folds <- read_rds('data/folds_indexes.rds')

#Set random seed
set.seed(1234)

#Make vector for weighting
p <- seq(from = 0, to = 1, by = 0.01)

#Save target, lsvm and pda predictions separetely
target <- ifelse(train$target == 'Y', 1, 0)
pda_ <- train$pda_
lsvm_ <- train$lsvm_

#Use cross validation to select best p for weighted average predictions
result_avg <- foreach(p1 = p, .combine = bind_rows) %dopar% {
   res <- foreach(j = 1:5, .combine = bind_rows) %dopar% {
     temp_ <- p1*pda_[folds[[j]]] + (1 - p1)*lsvm_[folds[[j]]]
     auc_score <- AUC(y_pred = temp_, y_true = target[folds[[j]]])
     data.frame(p_ = p1, auc_ = auc_score)
   }
   res
}
result_avg
res_avg <- result_avg %>%
  group_by(p_) %>%
  summarise(mean_auc = mean(auc_)) %>%
  arrange(desc(mean_auc))

#Use cross validation to select best p for weighted average predictions
result_geommean <- foreach(p1 = p, .combine = bind_rows) %dopar% {
  res <- foreach(j = 1:5, .combine = bind_rows) %dopar% {
    temp_ <- (pda_[folds[[j]]])^(1/p1) * (lsvm_[folds[[j]]])^(1/(1 - p1))
    auc_score <- AUC(y_pred = temp_, y_true = target[folds[[j]]])
    data.frame(p_ = p1, auc_ = auc_score)
  }
  res
}
result_geommean
res_geom <- result_geommean %>%
  group_by(p_) %>%
  summarise(mean_auc = mean(auc_)) %>%
  arrange(desc(mean_auc))

#Use cross validation to select best p for weighted harmonic mean
result_harm <- foreach(p1 = p, .combine = bind_rows) %dopar% {
  res <- foreach(j = 1:5, .combine = bind_rows) %dopar% {
    temp_ <- (p1 + (1 - p1)) / (p1/pda_[folds[[j]]] + (1 - p1)/lsvm_[folds[[j]]]) 
    auc_score <- AUC(y_pred = temp_, y_true = target[folds[[j]]])
    data.frame(p_ = p1, auc_ = auc_score)
  }
  res
}
result_harm
res_harm <- result_harm %>%
  group_by(p_) %>%
  summarise(mean_auc = mean(auc_)) %>%
  arrange(desc(mean_auc))

#Exploring results
summary(res_avg$mean_auc)
summary(res_geom$mean_auc)
summary(res_harm$mean_auc)

#Make weighted average prediction
pred_avg <- 0.47 * test$pda_ + (1 - 0.47) * test$lsvm_
summary(pred_avg)

#Submit prediction
submission$target <- pred_avg

submission %>% write_csv('results/weighted_avg.csv')
