library(tidyverse)
library(e1071)
library(Matrix)
library(caret)
library(MLmetrics)
library(future)
library(furrr)

train <- read_rds('results/train_new.rds')
test <- read_rds('results/test_new.rds')
submission <-read_csv('data/sample_submission.csv')

x_train <- train %>% select(-id, -target)
y_train <- train$target

x_test <- test %>% select(-id)

#Define parameters for svm tuning
cost_ <- 10^(-5:5)
gamma_ <- 10^(-5:-1)
kernels <- c("radial", "sigmoid")

set.seed(1234)
#Define folds
folds <- createFolds(y = as.factor(y_train), k = 10, list = TRUE)
folds

#Tuning svm parameters
plan(multiprocess)
time1 <- Sys.time()
result_tuning <- foreach(j = 1:10, .combine = bind_rows) %dopar% {
  kernel_res <- foreach(k = kernels, .combine = bind_rows) %dopar% {
    cost_res <- foreach(co = cost_, .combine = bind_rows) %dopar% {
      gamma_res <- foreach(ga = gamma_, .combine = bind_rows) %dopar% {
        x_train_temp <- Matrix(as.matrix(x_train[-folds[[j]],]), sparse = TRUE)
        y_train_temp <- y_train[-folds[[j]]]
        x_test_temp <- Matrix(as.matrix(x_train[folds[[j]],]), sparse = TRUE)
        y_test_temp <- y_train[folds[[j]]]
        
        temp_svm_model <- svm(x = x_train_temp, y = as.factor(y_train_temp),
                              kernel = k, cost = co, gamma = ga, 
                              class.weights = c("0" = 0.36, "1" = 0.64),
                              probability = TRUE, scale = TRUE)
        y_pred_temp <- predict(temp_svm_model, x_test_temp, probability = TRUE)
        y_pred_temp <- attr(y_pred_temp, "probabilities")[,1]
        auc_score <- AUC(y_pred = y_pred_temp, y_true = y_test_temp) 
        data.frame(fold = j, kernel = k, cost = co, gamma = ga, auc_score = auc_score)
      }
      gamma_res
    }
    cost_res
  }
  kernel_res
}
Sys.time() - time1
head(result_tuning)
result_tuning_average <- result_tuning %>%
  group_by(kernel, cost, gamma) %>%
  summarise(mean_auc_score = mean(auc_score),
            sd_auc_score = sd(auc_score)) %>% 
  arrange(desc(mean_auc_score))

p <- ggplot(result_tuning_average, aes(x = cost, y = mean_auc_score, color = kernel)) +
  geom_point() +
  geom_line()
p

p <- ggplot(result_tuning_average, aes(x = gamma, y = mean_auc_score, color = kernel)) +
  geom_point() +
  geom_line()
p

head(result_tuning_average)

tail(result_tuning_average)

#Set selected parameters
kernel <- "sigmoid"
cost <- 0.01
gamma <- 0.001

#create train and test matrix
x_train_matrix <- Matrix(as.matrix(x_train), sparse = TRUE)
x_test_matrix <- Matrix(as.matrix(x_test), sparse = TRUE)

#train svm 
model_svm <- svm(x = x_train_matrix, y = as.factor(y_train),
                 kernel = kernel, cost = cost, gamma = gamma, 
                 class.weights = c("0" = 0.36, "1" = 0.64), scale = TRUE,
                 probability = TRUE)

#make prediction
y_pred_test <- predict(temp_svm_model, x_test_matrix, probability = TRUE)
y_pred_test <- attr(y_pred_test, "probabilities")[,"1"]

#look at summary of predictions
summary(y_pred_test)
qplot(y_pred_test, geom = 'density')

submission$target <- y_pred_test

#save predictions
write_csv(submission, "results/model_svm2.csv")
