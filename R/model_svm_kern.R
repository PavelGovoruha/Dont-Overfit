library(tidyverse)
library(kernlab)
library(caret)
library(foreach)
library(MLmetrics)
library(future)
library(furrr)

train <- read_rds('results/train_new.rds')
test <- read_rds('results/test_new.rds')
submission <-read_csv('data/sample_submission.csv')

x_train <- train %>% select(-id, -target)
y_train <- train$target

x_test <- test %>% select(-id)

#Set random seed
set.seed(1234)

#Create folds
#Define folds
folds <- createFolds(y = as.factor(y_train), k = 5, list = TRUE)
folds

#Set parameters to tune ksvm
rbfdot <- 'rbfdot'
C_ <- seq(from = 100, to = 250, by = 10)

#Tuning svm parameters
plan(multiprocess)
time1 <- Sys.time()
result_tuning <- foreach(j = 1:5, .combine = bind_rows) %dopar% {
    cost_res <- foreach(co = C_, .combine = bind_rows) %dopar% {
        x_train_temp <- as.matrix(x_train[-folds[[j]],])
        y_train_temp <- y_train[-folds[[j]]]
        x_test_temp <- as.matrix(x_train[folds[[j]],])
        y_test_temp <- y_train[folds[[j]]]
        
        temp_svm_model <- ksvm(x = x_train_temp, y = as.factor(y_train_temp),
                              kernel = rbfdot, C = co, 
                              class.weights = c("0" = 36, "1" = 64), 
                              prob.model = TRUE,
                              scale = TRUE)
        y_pred_temp <- predict(temp_svm_model, x_test_temp, type = 'probabilities')
        y_pred_temp <- y_pred_temp[,2]
        auc_score <- AUC(y_pred = y_pred_temp, y_true = y_test_temp) 
        data.frame(fold = j, kernel = rbfdot, C = co, auc_score = auc_score)
    }
    cost_res
}
Sys.time() - time1

head(result_tuning)
result_tuning_average <- result_tuning %>%
  group_by(kernel, C) %>%
  summarise(mean_auc_score = mean(auc_score),
            sd_auc_score = sd(auc_score)) %>% 
  arrange(desc(mean_auc_score))

p <- ggplot(result_tuning_average, aes(x = C, y = mean_auc_score, color = kernel)) +
  geom_point() +
  geom_line()
p

head(result_tuning_average)
tail(result_tuning_average)

#Set selected parameters
kernel <- rbfdot
C_ <- 100


#train ksvm
#create train and test matrix
x_train_matrix <- as.matrix(x_train)
x_test_matrix <- as.matrix(x_test)

ksvm_model <- ksvm(x = x_train_matrix, y = as.factor(y_train),
                      kernel = rbfdot, C = C_, 
                      class.weights = c("0" = 36, "1" = 64), 
                      prob.model = TRUE,
                      scale = TRUE)
y_pred_test <- predict(ksvm_model, x_test_matrix, type = 'probabilities')
y_pred_test <- y_pred_test[,2]

#look at summary of predictions
summary(y_pred_test)
qplot(y_pred_test, geom = 'density')

submission$target <- y_pred_test

#save predictions
write_csv(submission, "results/model_svm_kern.csv")
