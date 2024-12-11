library(FNN)
library(xgboost)
library(mltools)
library(data.table)
library(randomForest)
library(plyr)
library(caret)
library(proxy)
library(mclust)
library(MASS)


assign_test_clusters <- function(test_similarity_matrix, training_cluster_labels) {
  num_test_samples <- nrow(test_similarity_matrix)
  num_training_samples <- length(training_cluster_labels)
  label_list<-unique(training_cluster_labels)
  ave_similarities<-matrix(0,num_test_samples,length(label_list))
  for (j in seq_along(label_list)) {
    cluster_indices <- which(training_cluster_labels == label_list[j])  ## if label_list is just 1-k then can just use j
    ave_similarities[,j]<-rowMeans(test_similarity_matrix[,cluster_indices])      
  }
  assigned_cluster_labels<-apply(ave_similarities,1,which.min)
  
  return(assigned_cluster_labels)
}

compare_ARI<-function(spat_sim,npcn,data){
  ### use hierarchical clustering and compute ARI 
  hc = hclust(as.dist(1 - spat_sim), method = "average")
  groups <- cutree(hc, k = npcn)
  ARI1 = adjustedRandIndex(as.factor(groups), as.factor(data$cluster_train))
  
  groups_test2 = assign_test_clusters(test_similarity_matrix = 1 - test_sim[-valid.index,],training_cluster_labels = groups)
  ARI2 = adjustedRandIndex(as.factor(groups_test2), as.factor(data$cluster_test[-valid.index]))
  
  groups_valid2 = assign_test_clusters(test_similarity_matrix = 1 - test_sim[valid.index,],training_cluster_labels = groups)
  ARI3 = adjustedRandIndex(as.factor(groups_valid2), as.factor(data$cluster_test[valid.index]))
  result = c(ARI1,ARI2,ARI3)
  return(result)
}

compare_xgb_MSE<-function(dtrain,dtest,dvalid){
  ### Fits an xgboost model on the data and records MSE.
  params <-
    list(
      booster = "gbtree",
      objective = "reg:squarederror",
      eta = 0.05,
      max_depth = 3,
      subsample = 1,
      eval_metric = "rmse"
    )
  
  xgb_sim<-
    xgb.train (
      params = params,
      data = dtrain,
      nrounds = 2500,
      watchlist = list(val = dvalid, train = dtrain),
      print_every_n = 50,
      maximize = F
    )
  index1 = which.min(xgb_sim$evaluation_log$val_rmse)
  xgb_sim <-
    xgb.train (
      params = params,
      data = dtrain,
      nrounds = index1,
      watchlist = list(val = dvalid, train = dtrain),
      print_every_n = 50,
      maximize = F
    )
  result1 = c(mean((predict(xgb_sim1, dtrain) - getinfo(dtrain,"label")) ^ 2),
              mean((predict(xgb_sim1, dvalid) - getinfo(dvalid,"label")) ^ 2),
              mean((predict(xgb_sim1, dtest) - getinfo(dtest,"label")) ^ 2))
  return(result1)
  
}


