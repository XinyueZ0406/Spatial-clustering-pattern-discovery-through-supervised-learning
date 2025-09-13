# Spatial-clustering-pattern-discovery-through-supervised-learning
## Introduction
This is an implementation of the paper "Spatial clustering pattern discovery through supervised learning".
Spatial patterns and relationships are crucial for statistical modeling and inference across various fields. This study develops a novel approach using supervised random forest to compute similarity scores between locations, effectively capturing spatial dependencies of a response variable. The approach begins by enriching location coordinates, enabling random forest to split space into irregular shaped subspaces. The similarity score is then derived from the proportion of trees in which two locations fall in the same node for the same values of other predictors. From the resulting similarity matrix, eigen-scores and cluster labels are extracted and integrated into predictive models such as XGBoost, GWR, and random forest. Two simulations and real data sets (house data and ocean data) indicate that the similarity matrix can capture more spatial information and significantly enhances the predictive performance of models than competing methods.
## Usage
Here, I gave an example of simulating data with two dimensional coordinates using selected parameters.
```
run runsims(ntrain=5000,ntest=1000,nvalid=1000,nclust=10,num_eigen=10,npred=8,ntr=200,nps=50,ndir=18,npcn=10)
```
Note: 
  1. ntrain: the number of training data
  2. ntest: the number of test data
  3. nvalid: the number of validation data
  4. nclust: the number of true clusters
  5. num_eigen: the number of eigen-scores used in fitting XGBoost model
  6. npred: the number of predictors
  7. ntr: the number of trees in random forest (N)
  8. nps: the number of pseudo observations (P)
  9. ndir: the number of enriched coordinates (M)
  10. npcn: the number of predicted clusters
## Ocean data
Here, I also provided the full workflow (ocean_data.R) for our ocean clustering analysis. Specifically, it includes:
  1. Data preprocessing: Load and clean the ocean climatology dataset.
  2. RF_Sim computation: Construct the random forest–based similarity matrix from ocean features.
  3. Clustering: Apply hierarchical clustering on the RF_Sim matrix to obtain cluster labels.
  4. Eigen-scores: Perform eigen-decomposition on the similarity matrix to extract eigen-scores that summarize spatial structure.
  5. Prediction with XGBoost: Train XGBoost models using ocean features combined with either cluster labels or eigen-scores, and evaluate predictive  performance.
  6. Visualization:
     (i) Plot spatial clustering results on a world map.
     (ii) Plot selected eigen-scores to visualize spatial patterns.
