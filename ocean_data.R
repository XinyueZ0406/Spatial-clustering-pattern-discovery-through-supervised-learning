library(MASS)
library(dplyr)
library(tibble)
library(xgboost)
library(mltools)
library(data.table)
library(randomForest)
library(plyr)
library(FNN)
library(caret)

make_enriched_coords_3d<-function(ndir){
  sigma=matrix(0,3,3)
  diag(sigma)=1
  rp=mvrnorm(n=ndir,rep(0,3),sigma)
  rp_n=t(apply(rp, 1, function(x) (x/sqrt(sum(x^2)))))
  return(rp_n)
}

spatial_RF_similarity_3d<-function(XYZcoord,X,Y,ntr,nps,ndir,stratified = FALSE,strata_col){
  ### XYZ coord is the unique 3-dimensional coordinates of the data points
  ### X is the predictor values
  ### Y is the response variable
  ### ntr is the number of trees
  ### nps is the number of pseudopoints.
  ### ndir is the number of additional coordinate directions.
  ### stratified: if TRUE, sample within strata defined by 'strata_col'.
  ### strata_col : column name (character) in X for stratification. 
  
  
  npred<-dim(X)[2]
  ##enrich coordinates
  rp_n=make_enriched_coords_3d(ndir)
  newFeatures_all<-  as.matrix(XYZcoord) %*% t(rp_n)
  XYZcoord_unq <- XYZcoord %>%
    rownames_to_column(".rid") %>%          
    select(.rid, x, y, z) %>%
    distinct(x, y, z, .keep_all = TRUE) %>% 
    column_to_rownames(".rid")    
  newFeatures<-  as.matrix(XYZcoord_unq) %*% t(rp_n)
  ndat<-dim(newFeatures)[1]
  ##    colnames(newFeatures) = paste0("d", seq(0, cr_num - 1, 1), "d")
  #set.seed(1)
  rf = randomForest(Y ~ ., data = cbind(X,as.data.frame(newFeatures_all),"Y"=Y), ntree = ntr, nodesize=30)
  similarity.matrix = matrix(0, ncol = ndat, nrow = ndat)
  ##choose pseudo observations
  #set.seed(1)
  if (isTRUE(stratified)) {
    strata_vals <- if (is.data.frame(X)) X[[strata_col]] else as.data.frame(X)[[strata_col]]
    idx_list <- split(seq_len(dim(X)[1]), strata_vals)
    pseudo_obs <- unlist(lapply(idx_list, function(idx) {
      sample(idx, nps)
    }), use.names = FALSE)
  } else {
    pseudo_obs <- sample(seq_len(dim(X)[1]), nps)
  }
  pseudo_data <- X[pseudo_obs,]
  ##compute similarity matrix
  for (i in seq_len(nps)) {
    vals <- unlist(pseudo_data[i, , drop = TRUE])
    mat  <- matrix(rep(vals, each = ndat), nrow = ndat, ncol = length(vals), byrow = FALSE)
    colnames(mat) <- colnames(pseudo_data)
    
    pred=predict(rf,newdata=cbind(mat,as.data.frame(newFeatures)),nodes=TRUE)
    node=t(attr(pred,"nodes"))
    similarity.matrix_new = apply(node, 2, function(x) colSums(x == node))
    similarity.matrix = similarity.matrix + similarity.matrix_new
  }
  ans<-list("train_coords"=XYZcoord,
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
  npred<-object$npred
  
  newdata_unq <- newdata %>%
    rownames_to_column(".rid") %>%          
    select(.rid, x, y, z) %>%
    distinct(x, y, z, .keep_all = TRUE) %>% 
    column_to_rownames(".rid")    
  newFeatures<-as.matrix(newdata_unq) %*% t(object$rp_n)
  ntest<-dim(newFeatures)[1]
  similarity.matrix = matrix(0, ncol = ntest, nrow = ndat,dimnames = list(rownames(object$newFeatures),rownames(newFeatures)))
  for (i in seq_len(nps)) {
    vals <- unlist(object$pseudo_data[i, , drop = TRUE])
    mat1 <- matrix(rep(vals, each = ndat), nrow = ndat, ncol = length(vals), byrow = FALSE)
    colnames(mat1) <- colnames(object$pseudo_data)
    mat2 <- matrix(rep(vals, each = ntest), nrow = ntest, ncol = length(vals), byrow = FALSE)
    colnames(mat2) <- colnames(object$pseudo_data)
    
    pred_train=predict(object$forest,newdata=cbind(mat1,as.data.frame(object$newFeatures)),nodes=TRUE)
    pred=predict(object$forest,newdata=cbind(mat2,as.data.frame(newFeatures)),nodes=TRUE)
    
    node_train=t(attr(pred_train,"nodes"))
    node=t(attr(pred,"nodes"))
    similarity.matrix_new = apply(node, 2, function(x) colSums(x == node_train))
    similarity.matrix = similarity.matrix + similarity.matrix_new
  }
  return(t(similarity.matrix)/(nps*object$ntree))
}


assign_test_clusters <- function(test_similarity_matrix, training_cluster_labels, input_is_distance = TRUE) {
  # remove NA labels if any
  label_list <- unique(training_cluster_labels[!is.na(training_cluster_labels)])
  num_test_samples <- nrow(test_similarity_matrix)
  ave_similarities <- matrix(NA_real_, nrow = num_test_samples, ncol = length(label_list))
  
  for (j in seq_along(label_list)) {
    cluster_indices <- which(training_cluster_labels == label_list[j])
    if (length(cluster_indices) == 0L) {
      ave_similarities[, j] <- NA_real_
    } else {
      ave_similarities[, j] <- rowMeans(test_similarity_matrix[, cluster_indices, drop = FALSE])
    }
  }
  
  # if input is distance, pick the MIN average; if similarity, pick the MAX
  if (input_is_distance) {
    best_idx <- max.col(-ave_similarities, ties.method = "first")  # which.min
  } else {
    best_idx <- max.col( ave_similarities, ties.method = "first")  # which.max
  }
  assigned_cluster_labels <- label_list[best_idx]
  return(assigned_cluster_labels)
}



# Read the ocean climatology dataset directly from the GitHub repository
# Source: https://github.com/brorfred/ocean_clustering
# (use the "Raw" link for the CSV file in that repo)
od <- read.csv("https://raw.githubusercontent.com/brorfred/ocean_clustering/main/data/tabulated_geospatial_montly_clim_360_720_ver_0_2_5.csv")

#remove the columns with NaN values
od=na.omit(od)
rownames(od) <- od$X
od=od[sample(nrow(od),3000),1:13]

#convert to three-dimension
cols=c("lat","lon")
od[cols]=lapply(od[cols],function(x) x*pi/180)

od$x=cos(od$lat)*cos(od$lon)
od$y=cos(od$lat)*sin(od$lon)
od$z=sin(od$lat)


#get unique locations
locations_uniq <- od %>% select("x","y","z") %>% distinct
set.seed(123)
#split data into train and test based on unique locations
train_loc <- locations_uniq[sample(nrow(locations_uniq),30000),]
test_loc <- setdiff(locations_uniq,train_loc)
train_list<- inner_join(od, train_loc, by = c("x","y","z"))$X

all_names=c("month","SST","PAR","mld","wind","EKE","bathymetry")
XX_train=od[which(od$X %in% train_list),all_names]
XX_test=od[which(!(od$X %in% train_list)),all_names]

XX_train_loc=od[which(od$X %in% train_list),c("x","y","z")]
XX_test_loc=od[which(!(od$X %in% train_list)),c("x","y","z")]

YY_train=od[which(od$X %in% train_list),'Chl']
YY_test=od[which(!(od$X %in% train_list)),'Chl']

spat_sim_model=spatial_RF_similarity_3d(XYZcoord=XX_train_loc,X=XX_train,Y=YY_train,
                                        ntr=200,nps=100,ndir=1000,stratified =TRUE,strata_col='month')

s<-spat_sim_model$similarity

s_t<-predict(spat_sim_model,newdata=XX_test_loc)

##use hierarchical clustering to get cluster labels
hc = hclust(as.dist(1 - s), method = "average")
ocean_groups <- cutree(hc, k = 55)
ocean_groups_test <- assign_test_clusters(test_similarity_matrix = 1 - s_t,
                                       training_cluster_labels = ocean_groups,
                                       input_is_distance = TRUE)

all_loc_train_g=cbind(XX_train_loc[rownames(XX_train_loc) %in% rownames(s),],g=as.factor(ocean_groups))
all_loc_test_g=cbind(XX_test_loc[rownames(XX_test_loc) %in% rownames(s_t),],g=as.factor(ocean_groups_test))

all_names2=c("month","SST","PAR","mld","wind","EKE","bathymetry","x","y","z","Chl")

train_all_x=inner_join(od[which(od$X %in% train_list),all_names2], all_loc_train_g, by = c("x","y","z"))
test_all_x=inner_join(od[which(!(od$X %in% train_list)),all_names2], all_loc_test_g, by = c("x","y","z"))

all_data=rbind(train_all_x,test_all_x)
all_data_oh=one_hot(as.data.table(all_data))

x_train=all_data_oh[1:nrow(train_all_x),]
x_test=all_data_oh[(nrow(train_all_x)+1):nrow(all_data_oh),]

all_names3=setdiff(colnames(x_train), c("x","y","z","Chl"))
## fit XGBoost with ocean features+ cluster labels
dtrain <- xgb.DMatrix(data = as.matrix(x_train[, ..all_names3]), label= as.matrix(log(x_train$Chl)))
dtest <- xgb.DMatrix(data = as.matrix(x_test[, ..all_names3]), label= as.matrix(log(x_test$Chl)))

params <- list(booster = "gbtree", objective = "reg:squarederror", eta=0.02, max_depth=12, 
               subsample=0.7,eval_metric="rmse")
xgb <- xgb.train (params = params, data = dtrain, nrounds = 20000,
                  watchlist = list(val=dtest,train=dtrain), print_every_n = 200,
                  early_stopping_rounds=200, maximize = F )

##use MDS to get eigen-scores
eig_s=eigen(s)
u_25 = eig_s$vectors[, 1:25]
ocean_ev = s %*% u_25
ocean_ev_test  = s_t %*% u_25

all_loc_train_eig=cbind(XX_train_loc[rownames(XX_train_loc) %in% rownames(s),],ocean_ev)
colnames(all_loc_train_eig)[4:ncol(all_loc_train_eig)]=paste0("eig_",seq(1,25,1))
all_loc_test_eig=cbind(XX_test_loc[rownames(XX_test_loc) %in% rownames(s_t),],ocean_ev_test)
colnames(all_loc_test_eig)[4:ncol(all_loc_test_eig)]=paste0("eig_",seq(1,25,1))

train_all_x=inner_join(od[which(od$X %in% train_list),all_names2], all_loc_train_eig, by = c("x","y","z"))
test_all_x=inner_join(od[which(!(od$X %in% train_list)),all_names2], all_loc_test_eig, by = c("x","y","z"))


## fit XGBoost with ocean features+  eigen-scores
all_names4=setdiff(colnames(train_all_x), c("x","y","z","Chl"))
dtrain <- xgb.DMatrix(data = as.matrix(train_all_x[, all_names4]), label= as.matrix(log(train_all_x$Chl)))
dtest <- xgb.DMatrix(data = as.matrix(test_all_x[, all_names4]), label= as.matrix(log(test_all_x$Chl)))

params <- list(booster = "gbtree", objective = "reg:squarederror", eta=0.02, max_depth=12, 
               subsample=0.7,eval_metric="rmse")
xgb <- xgb.train (params = params, data = dtrain, nrounds = 20000,
                  watchlist = list(val=dtest,train=dtrain), print_every_n = 200,
                  early_stopping_rounds=200, maximize = F )

##plot spatial clustering 

library(ggOceanMaps)
library(ggplot2)
library(ggmap)
library(ggalt)
library(RColorBrewer)
library(dplyr)
getPalette = colorRampPalette(brewer.pal(9, "Set1"))

plot_data_g=as.data.frame(cbind(lat=asin(all_loc_train_g$z),lon= atan2(all_loc_train_g$y, all_loc_train_g$x),
                  g=as.factor(ocean_groups)))
cols=c("lat","lon")
plot_data_g[cols]=lapply(plot_data_g[cols],function(x) x*180/pi)
basemap(limits = c(-180, 180, -90, 90),
        land.col = "#eeeac4", land.border.col = NA)+
  geom_point(data=plot_data_g,aes(x=lon, y=lat,color=as.factor(g)),size=1)+
  scale_color_manual(values = sample(getPalette(55)))+theme(legend.position="none")

##plot first eigen-scores
plot_data_es=as.data.frame(cbind(lat=asin(all_loc_train_eig$z),lon= atan2(all_loc_train_eig$y, all_loc_train_eig$x),
                                ocean_ev))
colnames(plot_data_es)[3:ncol(plot_data_es)]=paste0("eig_",seq(1,25,1))
plot_data_es[cols]=lapply(plot_data_es[cols],function(x) x*180/pi)
basemap(limits = c(-180, 180, -90, 90),
        land.col = "#eeeac4", land.border.col = NA)+
  geom_point(data=plot_data_es,aes(x=lon, y=lat,color=eig_1))+scale_color_gradient(
    low = "blue", high = "red") +
  labs(x = "Longitude", y = "Latitude") 



