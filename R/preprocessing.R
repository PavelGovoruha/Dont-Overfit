#Load libraries
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
                   number = 25,
                   verbose = FALSE,
                   allowParallel = TRUE,
                   rerank = TRUE,
                   returnResamp = "final",
                   saveDetails = TRUE)
rfFuncs$summary <- twoClassSummary

plan(multiprocess)
time1 <- Sys.time()
vars_rf <- rfe(x = train[,-c(1,2)], y = train$target, metric = 'ROC',
               rfeControl = ctrl_rfFuncs)
Sys.time() - time1

print(vars_rf)
plot(vars_rf)
rfe_rf_selection <- predictors(vars_rf)
rfe_rf_selection

#Select features by nbFuncs
ctrl_nb<- rfeControl(
  functions = nbFuncs,
  method = "boot",
  number = 25,
  verbose = FALSE,
  allowParallel = TRUE,
  rerank = TRUE,
  returnResamp = "final",
  saveDetails = TRUE
)
nbFuncs$summary <- twoClassSummary

plan(multiprocess)
time1 <- Sys.time()
vars_nb <- rfe(x = train[,-c(1,2)], y = train$target, metric = 'ROC',
               rfeControl = ctrl_nb)
Sys.time() - time1

vars_nb

selectetion_nb <- predictors(vars_nb)
plot(vars_nb)

#Use Boruta to select variables
time1 <- Sys.time()
boruta_selection <- Boruta(x = train[,-c(1,2)], y = train$target, doTrace = 3, 
                           maxRuns = 500)
Sys.time() - time1

boruta_selection

plot(boruta_selection)

getSelectedAttributes(boruta_selection, withTentative = TRUE)

getSelectedAttributes(boruta_selection, withTentative = FALSE)

boruta_selection <-getSelectedAttributes(boruta_selection, withTentative = TRUE)

#Add mean and positive ratio
train$pos_ratio_ <- apply(train[,-c(1,2)], 1, 
                          function(x){mean(x > 0)})

test$pos_ratio_ <- apply(test[,-1], 1, 
                          function(x){mean(x > 0)})

train$mean_ <- apply(train[,-c(1,2)], 1, mean)
test$mean_ <- apply(test[,-c(1,2)], 1, mean)

#Create list of selected predictors
selectetion_nb
boruta_selection
rfe_rf_selection

#"maxmin","pos_ratio_","mean_","v33",  "v65",  "v117", "v217",  
#               "v39", "v91",  "v295", "v189", "v16","v228","v268","v73",
#               "v237","v199","v201")

variables <- c("pos_ratio_","mean_","v33",  "v65",  "v117", "v217",  
               "v39", "v91",  "v295", "v189", "v16","v228","v268","v73",
               "v237","v199","v201")

length(variables)

#Save selected variables
train_new <- train %>% select(id, target, variables)
test_new <- test %>% select(id, variables)

#Save new train and test datasets
write_rds(train_new, 'data/train_new.rds')
write_rds(test_new, 'data/test_new.rds')
