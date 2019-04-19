library(tidyverse)
library(stringr)
library(foreach)
library(gridExtra)
library(coin)
library(future)
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

#Let's explore distribution of several independent variables
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

#Let's apply permutation test to see if there significent difference in variables between classes 0 and 1
plan(multiprocess)
time1 <- Sys.time()
result <- foreach(j = new_names, .combine = bind_rows) %dopar% {
  f <- as.formula(str_c(j, "as.factor(target)", sep = " ~ "))
  p_t <- t.test(f, data = train)$p.value
  p_wilcox <- wilcox.test(f, data = train)$p.value
  data.frame(variable = j, Wilcox = p_wilcox, Welch = p_t)
}
Sys.time() - time1

head(result)
result %>% filter(Welch < 0.05)
result %>% filter(Wilcox < 0.05)

result %>% filter(Welch < 0.05, Wilcox < 0.05)

#Save vector of selected variables
selected_vars <- result %>% filter(Welch < 0.05, Wilcox < 0.05) %>% select(variable) %>% pull()
selected_vars[1:10]

#Let's prepare data for cluster analyses
train_scaled <- scale(train[,3:ncol(train)])
train[,3:ncol(train)] <- train_scaled

test_scaled <- scale(test[,2:ncol(test)])
test[,2:ncol(test)] <- test_scaled
head(train)
head(test)
#Apply kmeans clustering
plan(multiprocess)
time1 <- Sys.time()
res_clustering <- foreach(j = 1:15, .combine = bind_rows) %dopar% {
  set.seed(1234+j)
  temp_df <- bind_rows(train, test)
  tot_ <- kmeans(x =temp_df[,3:ncol(train)] %>%
                   select(-selected_vars), centers = j, iter.max = 1000, nstart = 10)$tot.withinss
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

#Let's build from 2 to 8 clusters
all_df <- bind_rows(train, test)
all_df$clust_2 <- kmeans(all_df[,3:ncol(all_df)] %>%
                           select(-selected_vars), centers = 2, iter.max = 1000, nstart = 10)$cluster
all_df$clust_3 <- kmeans(all_df[,3:ncol(all_df)] %>%
                           select(-selected_vars), centers = 3, iter.max = 1000, nstart = 10)$cluster
all_df$clust_4 <- kmeans(all_df[,3:ncol(all_df)] %>%
                           select(-selected_vars), centers = 4, iter.max = 1000, nstart = 10)$cluster
all_df$clust_5 <- kmeans(all_df[,3:ncol(all_df)] %>%
                           select(-selected_vars), centers = 5, iter.max = 1000, nstart = 10)$cluster
all_df$clust_6 <- kmeans(all_df[,3:ncol(all_df)] %>%
                           select(-selected_vars), centers = 6, iter.max = 1000, nstart = 10)$cluster
all_df$clust_7 <- kmeans(all_df[,3:ncol(all_df)] %>%
                           select(-selected_vars), centers = 7, iter.max = 1000, nstart = 10)$cluster
all_df$clust_8 <- kmeans(all_df[,3:ncol(all_df)] %>%
                           select(-selected_vars), centers = 8, iter.max = 1000, nstart = 10)$cluster

train <- all_df[1:nrow(train),]
names(train)
variable_clust <- c("clust_2", "clust_3", "clust_4", "clust_5", "clust_6", "clust_7", "clust_8")

plan(multiprocess)
time1 <- Sys.time()
res_ch <- foreach(j = variable_clust, .combine = bind_rows) %dopar% {
  chi <- train %>% 
    select(c(j, "target")) %>%
    table() %>%
    chisq.test()
  data.frame(variable = j, p_value = chi$p.value)  
} 
Sys.time() - time1

res_ch

new_train <- train %>% select(id, target, selected_vars, clust_4)
l <- nrow(train) + 1
a <- nrow(all_df)
test <- all_df[l:a,]

new_test <- test %>% select(id, target, selected_vars, clust_4)

#Let's take a short look at new datasets
glimpse(new_train)
glimpse(new_test)

#Save new train and test sets
write_rds(new_train, 'results/new_train.rds')
write_rds(new_test, 'results/new_test.rds')
