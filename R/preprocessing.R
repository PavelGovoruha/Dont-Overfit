library(tidyverse)
library(moments)
library(foreach)
library(caret)

#Read dataset
train <- read_csv('data/train.csv')
test <- read_csv('data/test.csv')

train$target <- ifelse(train$target == 0, 'N', 'Y')
#Rename variables
new_names <- foreach(j = names(train)[3:ncol(train)], .combine = c) %dor% {
  str_c("v", as.character(j), "")
}

names(train)[3:ncol(train)] <- new_names
names(test)[2:ncol(test)] <- new_names

#' Add Features
#'
#' @param data - data.frame
#' @param skip_cols - columns which not used
#'
#' @return - add to data.frame min, max, mean, ... to data.frame
add_features <- function(data, skip_cols)
{
  data$mean_ <- apply(data[,-skip_cols], 1, mean)
  data$median_ <- apply(data[,-skip_cols], 1, median)
  data$min_ <- apply(data[,-skip_cols], 1, min)
  data$max_ <- apply(data[,-skip_cols], 1, max)
  data$sd_ <- apply(data[,-skip_cols], 1, sd)
  data$skewness_ <- apply(data[,-skip_cols], 1, skewness)
  data$kurtosis_ <- apply(data[,-skip_cols], 1, kurtosis)
  data$iqr_ <- apply(data[,-skip_cols], 1, IQR)
  
  return(data)
}

train <- add_features(train, skip_cols = c(1,2))
test <- add_features(test, skip_cols = 1)

train$target <- factor(train$target)

#Select features by rfFuncs
set.seed(1234)
ctrl_rfFuncs <- rfeControl(functions = rfFuncs ,
                   method = "boot",
                   number = 100,
                   verbose = FALSE,
                   allowParallel = TRUE,
                   rerank = TRUE,
                   returnResamp = "final",
                   saveDetails = TRUE)
rfFuncs$summary <- twoClassSummary

vars_rf <- rfe(x = train[,-c(1,2)], y = train$target, metric = 'auc',
               rfeControl = ctrl_rfFuncs) 
plot(vars_rf)
vars_rf

selected_by_rf <- c("v33", "v65", "v117", "v217", "v91")
#Select features by nbFuncs
ctrl_nb<- rfeControl(
  functions = nbFuncs,
  method = "boot",
  number = 100,
  verbose = FALSE,
  allowParallel = TRUE,
  rerank = TRUE,
  returnResamp = "final",
  saveDetails = TRUE
)
nbFuncs$summary <- twoClassSummary

vars_nb <- rfe(x = train[,-c(1,2)], y = train$target, metric = 'ROC',
               rfeControl = ctrl_nb) 
plot(vars_nb)
vars_nb

selected_by_nb <- c("v33", "v65", "v217", "v117")

#Scale train and test sets
train_scaled <- scale(train[,3:ncol(train)])
test_scaled <- scale(test[,2:ncol(test)])

train[,3:ncol(train)] <- train_scaled
test[,2:ncol(test)] <- test_scaled

#Save selected with rfFuncs
train %>% select(id, target, selected_by_rf) %>% write_rds('data/train_sel_rf.rds')
test %>% select(id, selected_by_rf) %>% write_rds('data/test_sel_rf.rds')

#Save selected with nbFuncs
train %>% select(id, target, selected_by_nb) %>% write_rds('data/train_sel_nb.rds')
test %>% select(id, selected_by_nb) %>% write_rds('data/test_sel_nb.rds')

#Save all selected variablse
selected_all_vars <- unique(union_all(selected_by_rf, selected_by_nb))
train %>% select(id, target, selected_all_vars) %>% write_rds('data/train_sel_all.rds')
test %>% select(id, selected_all_vars) %>% write_rds('data/test_sel_all.rds')
