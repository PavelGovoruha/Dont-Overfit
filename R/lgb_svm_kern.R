library(tidyverse)

lgb_ <- read_csv('results/lgb_model2.csv')
svm_ <- read_csv('results/model_svm_kern.csv')

lgb_svm <- lgb_

lgb_svm$target <- (lgb_$target + svm_$target)/2

write_csv(lgb_svm, 'results/lgb_svm.csv')