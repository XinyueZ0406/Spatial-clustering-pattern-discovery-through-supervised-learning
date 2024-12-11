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

spatial_RF_similarity_2d<-function(XYcoord,X,Y,ntr,nps,ndir){
  ### XY coord is the 2-dimensional coordinates of the data points
  ### X is the predictor values
  ### Y is the response variable
  ### ntr is the number of trees
  ### nps is the number of pseudopoints.
  ### ndir is the number of additional coordinate directions
  
  ndat<-length(Y)
  npred<-dim(X)[2]
  ##enrich coordinates
  newFeatures<-  make_enriched_coords_2d(XYcoord,ndir)
  ##colnames(newFeatures) = paste0("d", seq(0, cr_num - 1, 1), "d")
  rf = randomForest(Y ~ ., data = cbind(X,newFeatures,"Y"=Y), ntree = ntr)
  similarity.matrix = matrix(0, ncol = ndat, nrow = ndat)
  ##choose pseudo observations
  pseudo_obs<-sample(seq_len(ndat), nps)
  pseudo_data<-X[pseudo_obs,]
  ##compute similarity matrix
  for (i in seq_len(nps)) {
    pred=predict(rf,newdata=cbind(matrix(rep(pseudo_data[i,],each=ndat),ndat,npred),newFeatures,"Y"=Y),nodes=TRUE)
    node=t(attr(pred,"nodes"))
    similarity.matrix_new = apply(node, 2, function(x) colSums(x == node))
    similarity.matrix = similarity.matrix + similarity.matrix_new
  }
  ans<-list("train_coords"=XYcoord,
            "pseudo_data"=pseudo_data,
            "ndat"=ndat,
            "ndir"=ndir,
            "newFeatures"=newFeatures,
            "nps"=nps,
            "npred"=npred,
            "similarity"=similarity.matrix/(nps*ntr),
            "ntree"=ntr,
            "forest"=rf)
  
  class(ans)="spatial_similarity_rf_2d"
  return(ans)
}

predict.spatial_similarity_rf_2d<-function(object,newdata){
  nps<-object$nps
  ndir<-object$ndir
  ndat<-object$ndat
  ntest<-dim(newdata)[1]
  npred<-object$npred
  
  newFeatures<-make_enriched_coords_2d(newdata,ndir)
  similarity.matrix = matrix(0, ncol = ntest, nrow = ndat)
  for (i in seq_len(nps)) {
    pred_train=predict(object$forest,newdata=cbind(matrix(rep(object$pseudo_data[i,],each=ndat),ndat,npred),object$newFeatures,"Y"=rep(1,ndat)),nodes=TRUE)
    pred=predict(object$forest,newdata=cbind(matrix(rep(object$pseudo_data[i,],each=ntest),ntest,npred),newFeatures),nodes=TRUE)
    
    node_train=t(attr(pred_train,"nodes"))
    node=t(attr(pred,"nodes"))
    similarity.matrix_new = apply(node, 2, function(x) colSums(x == node_train))
    similarity.matrix = similarity.matrix + similarity.matrix_new
  }
  return(t(similarity.matrix)/(nps*object$ntree))
}

spatial_RF_similarity_3d<-function(XYcoord,X,Y,ntr,nps,ndir){
  ### XY coord is the 3-dimensional coordinates of the data points
  ### X is the predictor values
  ### Y is the response variable
  ### ntr is the number of trees
  ### nps is the number of pseudopoints.
  ### ndir is the number of additional coordinate directions
  
  ndat<-length(Y)
  npred<-dim(X)[2]
  ##enrich coordinates
  rp_n=make_enriched_coords_3d(ndir)
  newFeatures<-  XYcoord %*% t(rp_n)
  ##    colnames(newFeatures) = paste0("d", seq(0, cr_num - 1, 1), "d")
  #set.seed(1)
  rf = randomForest(Y ~ ., data = cbind(X,newFeatures,"Y"=Y), ntree = ntr)
  similarity.matrix = matrix(0, ncol = ndat, nrow = ndat)
  ##choose pseudo observations
  #set.seed(1)
  pseudo_obs<-sample(seq_len(ndat), nps)
  pseudo_data<-X[pseudo_obs,]
  ##compute similarity matrix
  for (i in seq_len(nps)) {
    pred=predict(rf,newdata=cbind(matrix(rep(pseudo_data[i,],each=ndat),ndat,npred),newFeatures,"Y"=Y),nodes=TRUE)
    node=t(attr(pred,"nodes"))
    similarity.matrix_new = apply(node, 2, function(x) colSums(x == node))
    similarity.matrix = similarity.matrix + similarity.matrix_new
  }
  ans<-list("train_coords"=XYcoord,
            "pseudo_data"=pseudo_data,
            "ndat"=ndat,
            "ndir"=ndir,
            "newFeatures"=newFeatures,
            "rp_n"=rp_n,
            "nps"=nps,
            "npred"=npred,
            "similarity"=similarity.matrix/(nps*ntr),
            "ntree"=ntr,
            "forest"=rf)
  
  class(ans)="spatial_similarity_rf_3d"
  return(ans)
}

predict.spatial_similarity_rf_3d<-function(object,newdata){
  nps<-object$nps
  ndir<-object$ndir
  ndat<-object$ndat
  ntest<-dim(newdata)[1]
  npred<-object$npred
  
  newFeatures<-newdata %*% t(object$rp_n)
  similarity.matrix = matrix(0, ncol = ntest, nrow = ndat)
  for (i in seq_len(nps)) {
    pred_train=predict(object$forest,newdata=cbind(matrix(rep(object$pseudo_data[i,],each=ndat),ndat,npred),object$newFeatures,"Y"=rep(1,ndat)),nodes=TRUE)
    pred=predict(object$forest,newdata=cbind(matrix(rep(object$pseudo_data[i,],each=ntest),ntest,npred),newFeatures),nodes=TRUE)
    
    node_train=t(attr(pred_train,"nodes"))
    node=t(attr(pred,"nodes"))
    similarity.matrix_new = apply(node, 2, function(x) colSums(x == node_train))
    similarity.matrix = similarity.matrix + similarity.matrix_new
  }
  return(t(similarity.matrix)/(nps*object$ntree))
}
