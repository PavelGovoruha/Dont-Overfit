library(tidyverse)
library(stringr)
library(foreach)
library(moments)
library(gridExtra)
library(future)
library(corrplot)
library(furrr)
#Load data
train <- read_csv('data/train.csv')
test <- read_csv('data/test.csv')

#Take a look
glimpse(train)
glimpse(test)

#Check distiribution of target variable
table(train$target)
prop.table(table(train$target))

#Check data for missing values
sum(is.na(train))
sum(is.na(test))

#Let's rename variables 0-299
new_names <- foreach(j = names(train)[3:ncol(train)], .combine = c) %dopar% {
  str_c("v", as.character(j), "")
}
new_names[1:10]

names(train)[3:ncol(train)] <- new_names
names(test)[2:ncol(test)] <- new_names

names(train)
names(test)

#Let's explore distribution of several variables
for(j in 1:10){
   set.seed(2019 + j)
   var_ <- sample(new_names, 1)

   p1 <- train %>%
      ggplot(aes(x = (!!!rlang::syms(var_)))) +
      geom_histogram() +
      ggtitle(var_)

   p2 <- train %>%
      ggplot(aes(x = (!!!rlang::syms(var_)))) +
      geom_density() +
      ggtitle(var_)

   p <- grid.arrange(p1,p2)
      ggsave(filename = str_c(str_c("plots/", var_, collapse = "_"), ".jpeg", sep = ""), plot = p,
           device = 'jpeg')
}
#It looks like variable have similar distribution to normal

#Let's explore correlation between variables
corrplot(corr = cor(train[,3:ncol(train)]))

#There no highly correlated variables

#Let's see how mean per row distribted by target variable
mean_ <- apply(train[,3:ncol(train)], 1, mean)

p <- data.frame(mean_ = mean_, target = train$target) %>%
  ggplot() +
  geom_density(aes(x = mean_, color = factor(target))) +
  ggtitle('Distribution of mean per row')
p
ggsave(filename = 'plots/means_per_row.jpeg', plot = p, device = 'jpeg')
#The means looks quite different but both near the 0 

#Let's see how mean per column distributed
mean_ <- sapply(train[,3:ncol(train)], mean)
p <- data.frame(mean_ = mean_) %>%
  ggplot() +
  geom_density(aes(x = mean_)) +
  ggtitle('Distribution of mean per column')
p
ggsave(filename = 'plots/means_per_column.jpeg', plot = p, device = 'jpeg')

#Let's see how standard deviation distributed per row
sd_ <- apply(train[,3:ncol(train)], 1, sd)

p <- data.frame(sd_ = sd_, target = train$target) %>%
  ggplot() +
  geom_density(aes(x = sd_, color = factor(target))) +
  ggtitle('Distribution of sd per row')
p
ggsave(filename = 'plots/sd_per_row.jpeg', plot = p, device = 'jpeg')

#Let's see how standard deviation distributed per column
sd_ <- apply(train[,3:ncol(train)], 2, sd)

p <- data.frame(sd_ = sd_) %>%
  ggplot() +
  geom_density(aes(x = sd_)) +
  ggtitle('Distribution of sd per column')
p
ggsave(filename = 'plots/sd_per_column.jpeg', plot = p, device = 'jpeg')

#Let's see how skewness distributed per row
skewness_ <- apply(train[,3:ncol(train)], 1, skewness)

p <- data.frame(skewness_ = skewness_, target = train$target) %>%
  ggplot() +
  geom_density(aes(x = skewness_, color = factor(target))) +
  ggtitle('Distribution of skewness per row')
p
ggsave(filename = 'plots/skewness_per_row.jpeg', plot = p, device = 'jpeg')

#Let's see how skewness distributed per column
skewness_ <- apply(train[,3:ncol(train)], 2, skewness)

p <- data.frame(skewness_ = skewness_) %>%
  ggplot() +
  geom_density(aes(x = skewness_)) +
  ggtitle('Distribution of skewness per column')
p
ggsave(filename = 'plots/skewness_per_column.jpeg', plot = p, device = 'jpeg')

#Let's see how kurtosis distributed per row
kurtosis_ <- apply(train[,3:ncol(train)], 1, kurtosis)

p <- data.frame(kurtosis_ = kurtosis_, target = train$target) %>%
  ggplot() +
  geom_density(aes(x = kurtosis_, color = factor(target))) +
  ggtitle('Distribution of kurtosis per row')
p
ggsave(filename = 'plots/kurtosis_per_row.jpeg', plot = p, device = 'jpeg')

#Let's see how kurtosis distributed per column
kurtosis_ <- apply(train[,3:ncol(train)], 2, kurtosis)

p <- data.frame(kurtosis_ = kurtosis_) %>%
  ggplot() +
  geom_density(aes(x = kurtosis_)) +
  ggtitle('Distribution of kurtosis per column')
p
ggsave(filename = 'plots/kurtosis_per_column.jpeg', plot = p, device = 'jpeg')

#Let's apply statistical test to see if there significent difference in variables between classes 0 and 1
plan(multiprocess)
time1 <- Sys.time()
result <- foreach(j = new_names, .combine = bind_rows) %dopar% {
  f <- as.formula(str_c(j, "as.factor(target)", sep = " ~ "))
  p_t <- t.test(f, data = train, var.equal = TRUE)$p.value
  p_wilcox <- wilcox.test(f, data = train)$p.value
  data.frame(variable = j, Wilcox = p_wilcox, Student = p_t)
}
Sys.time() - time1

head(result)
result %>% filter(Student < 0.05)
result %>% filter(Wilcox < 0.05)

result %>% filter(Student < 0.05, Wilcox < 0.05)

#Let's see how p-values are distributed
p <- result %>% ggplot() +
  geom_boxplot(aes(x = "Student t-test", y = Student)) +
  geom_boxplot(aes(x = "Wilcox test", y = Wilcox)) +
  ggtitle('Distribution of p-values of different test')
p
ggsave(filename = 'plots/p_values.jpeg', plot = p, device = 'jpeg')

#As we see for both tests it quit the same, we select variable which p-value of both test below 0.05

#Save vector of selected variables
selected_vars <- result %>% filter(Student < 0.05, Wilcox < 0.05) %>% select(variable) %>% pull()
selected_vars[1:10]
length(selected_vars)

#Let's prepare data for cluster analyses
train_scaled <- scale(train[,3:ncol(train)]) %>% as_data_frame()

test_scaled <- scale(test[,2:ncol(test)]) %>% as_data_frame()

#Apply kmeans clustering
plan(multiprocess)
time1 <- Sys.time()
res_clustering <- foreach(j = 1:15, .combine = bind_rows) %dopar% {
  set.seed(1234+j)
  temp_df <- bind_rows(train_scaled, test_scaled)
  tot_ <- kmeans(x =temp_df, centers = j, iter.max = 1000, nstart = 10)$tot.withinss
  cat(str_c(j, "iteration\n", sep = " "))
  data.frame(n_clusters = j, tot_withinss = tot_)
}
Sys.time() - time1

p <- res_clustering %>%
  ggplot() +
  geom_line(aes(x = n_clusters, y = tot_withinss), size = 0.6) +
  geom_point(aes(x = n_clusters, y = tot_withinss), size = 0.6) +
  xlab('Number of clusters') +
  ylab('Within-cluster sum of squares')
p
ggsave(filename = 'plots/kmeans.jpeg', plot = p, device = 'jpeg')

#As we can see there no clusters in data

#Let's see how ratio of positive values distributed per row
pos_ratio_ <- apply(train[,3:ncol(train)], 1, function(x){return(mean(x > 0))})

p <- data.frame(pos_ratio = pos_ratio_, target = train$target) %>%
  ggplot() +
  geom_density(aes(x = pos_ratio, color = factor(target))) +
  ggtitle('Distribution of ratio positive values per row')
p
ggsave(filename = 'plots/positve_ratio.jpeg', plot = p, device = 'jpeg')

#Let's how min values distributed per row
min_ <- apply(train[,3:ncol(train)], 1, min)

p <- data.frame(min_ = min_, target = train$target) %>%
  ggplot() +
  geom_density(aes(x = min_, color = factor(target))) +
  ggtitle('Distribution of min values per row')
p
ggsave(filename = 'plots/min_.jpeg', plot = p, device = 'jpeg')

#Let's see how max values distributed per row
max_ <- apply(train[,3:ncol(train)], 1, max)

p <- data.frame(max_ = max_, target = train$target) %>%
  ggplot() +
  geom_density(aes(x = max_, color = factor(target))) +
  ggtitle('Distribution of max values per row')
p
ggsave(filename = 'plots/max_.jpeg', plot = p, device = 'jpeg')

#Let's see how  IQR per row distributed
iqr_ <- apply(train[,3:ncol(train)], 1, IQR)

p <- data.frame(iqr_ = iqr_, target = train$target) %>%
  ggplot() +
  geom_density(aes(x = iqr_, color = factor(target))) +
  ggtitle('Distribution of IQR per row')
p
ggsave(filename = 'plots/iqr_.jpeg', plot = p, device = 'jpeg')

#Let's create function to add some new variables

#' Add features
#'
#' @param data - data.frame to transform 
#' @param sel_vars - names of previously selected variables
#' @param skip_cols - cols which will be skipped during transformation
#'
#' @return - transformed data.frame
add_features <- function(data, sel_vars, skip_cols){
  data$mean_ <- apply(data[,-skip_cols], 1, mean)
  data$sd_ <- apply(data[,-skip_cols], 1, sd)
  data$kurtosis_ <- apply(data[,-skip_cols], 1, kurtosis)
  data$skewness_ <- apply(data[,-skip_cols], 1, skewness)
  data$min_ <- apply(data[,-skip_cols], 1, min)
  data$max_ <- apply(data[,-skip_cols], 1, max)
  data$pos_ratio_ <- apply(data[,-skip_cols], 1, function(x){return(mean(x > 0))})
  data$iqr_ <- apply(data[,-skip_cols], 1, IQR)
  data_transformed <- data %>%
    select(sel_vars, skip_cols, mean_, sd_, kurtosis_, skewness_,
           min_, max_, pos_ratio_, iqr_)
  return(data_transformed)
}

#Let's transform train and test datasets
train_new <- add_features(data = train, sel_vars = selected_vars, 
                          skip_cols = c(1, 2))
test_new <- add_features(data = test, sel_vars = selected_vars,
                         skip_cols = 1)
#Take a look at new data
glimpse(train_new)
glimpse(test_new)

#Save new train and test sets
write_rds(train_new, 'results/train_new.rds')
write_rds(test_new, 'results/test_new.rds')
