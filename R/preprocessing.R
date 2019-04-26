library(tidyverse)
library(moments)
library(foreach)
library(future)
library(caret)
library(Boruta)

#Read dataset
train <- read_csv('data/train.csv')
test <- read_csv('data/test.csv')

train$target <- ifelse(train$target == 0, 'N', 'Y')

#Rename variables
new_names <- foreach(j = names(train)[3:ncol(train)], .combine = c) %dopar% {
  str_c("v", as.character(j), "")
}

names(train)[3:ncol(train)] <- new_names
names(test)[2:ncol(test)] <- new_names

train$target <- factor(train$target)

#Select features by rfFuncs
plan(multiprocess)
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

time1 <- Sys.time()
vars_rf <- rfe(x = train[,-c(1,2)], y = train$target, metric = 'ROC',
               rfeControl = ctrl_rfFuncs, sizes = 4:15)
Sys.time() - time1

print(vars_rf)
plot(vars_rf, type = 'l')

#Varibles selected by rfe, rfFuncs are v33, v65, v117, v217, v91

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

time1 <- Sys.time()
vars_nb <- rfe(x = train[,-c(1,2)], y = train$target, metric = 'ROC',
               rfeControl = ctrl_nb, sizes = 4:15)
Sys.time() - time1

vars_nb
plot(vars_nb)

#Variables selected by nbFuncs v33, v65, v217, v117

#Use Boruta to select variables
time1 <- Sys.time()
boruta_selection <- Boruta(x = train[,-c(1,2)], y = train$target, doTrace = 3, maxRuns = 500)
Sys.time() - time1

boruta_selection

plot(boruta_selection)

getSelectedAttributes(boruta_selection, withTentative = TRUE)

getSelectedAttributes(boruta_selection, withTentative = TRUE)

#Selected variables with Boruta v117, v189, v217, v33, v65

#' Add Features
#'
#' Add mean, max, min and others statistics per row
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

#Add some statistics per row
train <- add_features(data = train, skip_cols = c(1,2))
test <- add_features(data = test, skip_cols = 1)

#Scale train and test sets
train_scaled <- scale(train[,3:ncol(train)])
test_scaled <- scale(test[,2:ncol(test)])

train[,3:ncol(train)] <- train_scaled
test[,2:ncol(test)] <- test_scaled

#Create list of selected predictors
variables <- c("v33", "v65", "v117", "v217", "v91", "v189", "v116", "v214", "v295",
               "v17", "v39", "mean_", "median_", "min_", "max_", "skewness_", "kurtosis_", "iqr_")
variables

#Save selected with rfFuncs
train_new <- train %>% select(id, target, variables)
test_new <- test %>% select(id, variables)

#Save new train and test datasets
write_rds(train_new, 'data/train_new.rds')
write_rds(test_new, 'data/test_new.rds')
