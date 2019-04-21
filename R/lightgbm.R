library(tidyverse)
library(lightgbm)


#Read data
train <- read_rds('results/train_new.rds')
test <- read_rds('results/test_new.rds')
submission <-read_csv('data/sample_submission.csv')

x_train <- train %>% select(-id, -target) %>% as.matrix()
y_train <- train$target

x_test <- test %>% select(-id) %>% as.matrix()

#Create lgb datasets
dtrain <- lgb.Dataset(data = x_train, label = y_train)

#Set parameters
params <- list(objective = "binary", 
               boost="gbdt",
               metric="auc",
               num_threads= 4,
               learning_rate = 0.1,
               num_leaves = 24,
               max_depth = 6,
               feature_fraction = 0.3,
               bagging_freq = 5,
               bagging_fraction = 1,
               min_data_in_leaf = 10,
               min_sum_hessian_in_leaf = 0.1,
               lambda_l1 = 1,
               lambda_l2 = 1,
               verbosity = 1)

#set random seed
set.seed(1234)

#cross validation
lgb_cv <- lgb.cv(params,
                 data = dtrain,
                 nrounds=10000,
                 nfold = 10,
                 early_stopping_rounds = 300,
                 eval_freq=10,
                 seed=1234)
#best_score
lgb_cv$best_score

#best iteration
lgb_cv$best_iter

#train light gbm
best_iter <- lgb_cv$best_iter

set.seed(1234)
lgb_model <- lgb.train(params = params, data = dtrain, nrounds = best_iter, eval_freq = 10,
                       seed = 1234)

#feature importances
importances_ <- lgb.importance(lgb_model)
importances_

p <- importances_ %>%
  ggplot(aes(x = reorder(Feature,Gain), y = Gain, fill = Gain)) +
  geom_col() +
  coord_flip()
p
ggsave(filename = 'plots/lightgbm_feature_importances.jpeg',
       plot = p,
       device = 'jpeg')

#make prediction
pred_test <- predict(lgb_model, x_test)
pred_test

qplot(x = pred_test, geom = 'density')
length(pred_test)

summary(pred_test)
prop.table(table(as.numeric(pred_test > 0.64)))

#save prediction
submission$target <- pred_test
write_csv(submission, 'results/lgb_model3.csv')
