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

#Scale data
scaled_train <- scale(train[,-c(1,2)])
scaled_test <- scale(test[,-1])

train[,-c(1,2)] <- scaled_train
test[,-1] <- scaled_test

#Create list of selected predictors
variables <- c('v33', 'v65', 'v117', 'v217', 'v91')
variables

#Save selected with rfFuncs
train_new <- train %>% select(id, target, variables)
test_new <- test %>% select(id, variables)

#Save new train and test datasets
write_rds(train_new, 'data/train_new.rds')
write_rds(test_new, 'data/test_new.rds')

