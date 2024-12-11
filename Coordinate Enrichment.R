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

make_enriched_coords_2d<-function(XYcoord,ndir){
  transMatrix = matrix(NA, nrow = 2, ncol = ndir)
  transMatrix[1, ] = cos(pi / ndir * seq(0, ndir - 1))
  transMatrix[2, ] = sin(pi / ndir * seq(0, ndir - 1))
  return( XYcoord %*% transMatrix)
}

make_enriched_coords_3d<-function(ndir){
  sigma=matrix(0,3,3)
  diag(sigma)=1
  rp=mvrnorm(n=ndir,rep(0,3),sigma)
  rp_n=t(apply(rp, 1, function(x) (x/sqrt(sum(x^2)))))
  return(rp_n)
}

