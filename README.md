# Dont Overfit II
https://www.kaggle.com/c/dont-overfit-ii/overview

This is my solution for Kaggle Playground Competition 'Dont Overfit II' 

We have two datasets:

train - with 250 rows and 300 variables except id and target

with - 19750 rows and 300 variables except id

Our goal to predict probabilty of target variable

Variable selection was made by RFE (package 'caret') and Boruta algorithm

Selected variables are pos_ratio_, mean_, 33, 65, 117, v217,  
               v39, v91, v295, v189, v16, v228, 268, 73,
               237, v199, v201.

Where mean_ is mean of all variables per row;
      pos_ratio_ is $\frac{\sum_{i = 0}^{299} variable[i] > 0}{300}$ per row
               
I build two models :

1. Penalized Discriminant Analyse
2. Support Vector Machine with linear kernels

Final step is taking weighted average of these models

It gave public leaderboard score 0.861 and private leaderboard score 0.852
