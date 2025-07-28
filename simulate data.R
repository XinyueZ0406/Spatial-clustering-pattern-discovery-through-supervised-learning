simulate_data<-function(ncluster,ntrain,ntest,npred){
  x_center=runif(ncluster,0.2,0.8)
  y_center=runif(ncluster,0.2,0.8)
  mus=cbind(x_center,y_center)
  
  nd<-ntrain+ntest
  
  ##create data points
  xc=runif(nd,0,1)
  yc=runif(nd,0,1)
  coord=cbind(xc,yc)
  ##get true cluster label
  dd=dist(coord,mus)
  cluster=apply(dd,1,which.min)
  
  XX<-matrix(rnorm(nd*npred),nd,npred)
  beta0=round(rnorm(ncluster,mean=0,sd=1),2)
  betas = matrix(rnorm(ncluster * npred, mean = 0, sd = 1), nrow = ncluster, ncol = npred) 
  for (i in seq_len(ncluster)) {
    rid = sample(seq_len(npred), sample(1:3, 1))
    betas[i, rid] = 0
  }
  
  beta_full<-betas[as.numeric(as.factor(cluster)),]
  Y = rowSums(beta_full * XX)+beta0[as.numeric(as.factor(cluster))] + rnorm(nd)
  train.index <- createDataPartition(cluster, p = ntrain/nd, list = FALSE)
  return(list("coord_train"=coord[train.index,],
              "coord_test"=coord[-train.index,],
              "XX_train"=XX[train.index,],
              "XX_test"=XX[-train.index,],
              "cluster_train"=cluster[train.index],
              "cluster_test"=cluster[-train.index],
              "Y_train"=Y[train.index],
              "Y_test"=Y[-train.index],
              "Beta0"=beta0,   ## For debugging and comparison
              "Beta1"=betas    ## For debugging and comparison
  ))
}

runsims<-function(ntrain=5000,ntest=10000,nvalid=1000,simlist=NULL,nclust=10,num_eigen=10,
                  npred=8,ntr=tree_num,nps=p_num,ndir=cr_num,npcn=clust_num){
  
  data<-simulate_data(ncluster=nclust,ntrain=ntrain,ntest=ntest+nvalid,npred=8)
  valid.index <- createDataPartition(data$cluster_test, p = nvalid/(nvalid+ntest), list = FALSE)
  
  spat_sim_model<-spatial_RF_similarity_2d(data$coord_train,
                                           data$XX_train,
                                           data$Y_train,
                                           ntr=ntr,
                                           nps=nps,
                                           ndir=ndir)
  
  spat_sim<-spat_sim_model$similarity
  
  test_sim<-predict(spat_sim_model,newdata=data$coord_test)
  
  
  
  ##compute eigenvectors
  
  eig_s = eigen(spat_sim)
  u = eig_s$vectors[, seq_len(num_eigen)]
  
  train_sim_ev = spat_sim %*% u
  test_sim_ev  = test_sim %*% u

  ##fit xgboost model with features+eigenvectors
  
  dtrain <-
    xgb.DMatrix(data = as.matrix(cbind(data$XX_train,train_sim_ev)),
                label = as.matrix(data$Y_train))
  dvalid <-
    xgb.DMatrix(data = as.matrix(cbind(data$XX_test,test_sim_ev))[valid.index,],
                label = as.matrix(data$Y_test[valid.index]))
  
  dtest <-
    xgb.DMatrix(data = as.matrix(cbind(data$XX_test,test_sim_ev))[-valid.index,], label = as.matrix(data$Y_test[-valid.index]))
  
  result1 = compare_xgb_MSE(dtrain,dtest,dvalid)
  
  ##fit xgboost model with features+group label
  hc = hclust(as.dist(1 - spat_sim), method = "average")
  groups <- cutree(hc, k = nclust)

  ## compute ARI
  ARI1 = adjustedRandIndex(as.factor(groups), as.factor(data$cluster_train))
  groups_test2 = assign_test_clusters(test_similarity_matrix = 1 - test_sim[-valid.index,],training_cluster_labels = groups)

  ARI2 = adjustedRandIndex(as.factor(groups_test2), as.factor(data$cluster_test[-valid.index]))
  
  groups_valid2 = assign_test_clusters(test_similarity_matrix = 1 - test_sim[valid.index,],training_cluster_labels = groups)
  ARI3 = adjustedRandIndex(as.factor(groups_valid2), as.factor(data$cluster_test[valid.index]))
  
  
  train_all_sim3=cbind(as.data.frame(data$XX_train), as.factor(groups))
  colnames(train_all_sim3)[dim(train_all_sim3)[2]]='g'
  train_all_sim3 = one_hot(as.data.table(train_all_sim3))
  
  test_all_sim3=cbind(as.data.frame(data$XX_test[-valid.index,]), as.factor(groups_test2))
  colnames(test_all_sim3)[dim(test_all_sim3)[2]]='g'
  test_all_sim3 = one_hot(as.data.table(test_all_sim3))
  
  valid_all_sim3=cbind(as.data.frame(data$XX_test[valid.index,]), as.factor(groups_valid2))
  colnames(valid_all_sim3)[dim(valid_all_sim3)[2]]='g'
  valid_all_sim3 = one_hot(as.data.table(valid_all_sim3))
  
  dtrain <-
    xgb.DMatrix(data = as.matrix(train_all_sim3),
                label = as.matrix(data$Y_train))
  dtest <-
    xgb.DMatrix(data = as.matrix(test_all_sim3),
                label = as.matrix(data$Y_test[-valid.index]))
  dvalid <-
    xgb.DMatrix(data = as.matrix(valid_all_sim3),
                label = as.matrix(data$Y_test[valid.index]))
  
  result3 = compare_xgb_MSE(dtrain,dtest,dvalid)
  
  result = c(ARI1,ARI2,ARI3,result1,result3)
  print(result)
  return(list('data'=data,
              'valid.index'=valid.index,
              'spat_sim'=spat_sim,
              'test_sim'=test_sim,
              'groups_train'=groups,
              'groups_test'=groups_test2,
              'groups_valid'=groups_valid2,
              'result'=result))
}




