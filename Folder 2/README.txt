# Recommender Systems Final Project

By Brad Shook, Daniel Cowan, and Drew Dibble

This repository contains files relating to our final project for CSC 381 (Recommender Systems) at Davidson College. 
The repository is divided into three folders: Folder 1, Folder 2, and Folder 3. Folder 1 contains the LaTeX tex file and the 
files required to generate the PDF document. Folder 2 contains all python files and results, in addition to this README.txt file. 
Folder 3 contains the data used to train these models.

The recommender system is contained in Folder 2/recommendations.py. The file contains implementations for of 10 different algorithms: 
1) user-based collaborative filtering using Euclidean distance for user-user similarity calculations (UU-CF-D), 
2) user-based collaborative filtering using Pearson similarity (UU-CF-P), 
3) item-based collaborative filtering using Euclidean distance (II-CF-D), 
4) item-based collaborative filtering using Pearson similarity (II-CF-P), 
5) matrix factorization collaborative filtering with stochastic gradient descent (MF-SGD), 
6) matrix factorization collaborative filtering with alternating least squares (MF-ALS), 
7) TFIDF content-based method with cosine similarity for content similarity calculations (TFIDF), 
8) a hybrid model using a combination of the II-CF-D and TFIDF methods (HYB-D),
9) a hybrid model using a combination of the II-CF-P and TFIDF methods (HYB-P),
10) and a neural collaborative filtering model (NCF). Intructions for running each of these models are below:

NOTE: Folder 2/requirements.txt contains the pip packages and versions that were installed on my local machine when running this code. 
Specifically, one needs: os, numpy, scipy, statsmodels, pandas, csv, matplotlib, pickle, time, warnings, traceback, copy, sklearn, keras, 
and tensorflow. If there are any issues running the code, please contact us.

## Recommendations

#### UU-CF

In order to run UU-CF: 
1) Run RML or R to load in the data
2) Run the corresponding PD-RML or PD-R depending on which data you loaded in 1)
3) Run SIMU, read or write the user-user similarity matrix with a specificied similarity metric.
4) Run RECS, enter 'uu-cf' and answer the pop-up questions. When you are asked for the similarity calculation, enter 'd' for distance or 'p' for pearson. 

#### II-CF

In order to run II-CF: 
1) Run RML or R to load in the data
2) Run the corresponding PD-RML or PD-R depending on which data you loaded in 1)
3) Run SIM, read or write the item-item similarity matrix with a specificied similarity metric.
4) Run RECS, enter 'ii-cf' and answer the pop-up questions. When you are asked for the similarity calculation, enter 'd' for distance or 'p' for pearson.

#### MF-SGD

In order to run MF-SGD: 
1) Run RML or R to load in the data
2) Run the corresponding PD-RML or PD-R depending on which data you loaded in 1)
3) Run T, enter Y. This will create train and test splits.
4) Run MF-SGD. Follow the pop-up questions and continue.
5) Run RECS, enter 'mf-sgd' and answer the pop-up questions. This will return reccommendations for the given user.

#### MF-ALS

In order to run MF-ALS: 
1) Run RML or R to load in the data
2) Run the corresponding PD-RML or PD-R depending on which data you loaded in 1)
3) Run T, enter Y. This will create train and test splits.
4) Run MF-ALS. Follow the pop-up questions and continue.
5) Run RECS, enter 'mf-als' and answer the pop-up questions. This will return reccommendations for the given user.

#### TFIDF

In order to run TFIDF: 
1) Run RML or R to load in the data
2) Run the corresponding PD-RML or PD-R depending on which data you loaded in 1)
3) Run TFIDF.
5) Run RECS, enter 'tfidf' and answer the pop-up questions. This will return reccommendations for the given user.

#### HYB

In order to run HYB: 
1) Run RML or R to load in the data
2) Run the corresponding PD-RML or PD-R depending on which data you loaded in 1)
3) Run TFIDF.
4) Run SIM.
5) Run RECS, enter 'hybrid' and answer the pop-up questions. This will return reccommendations for the given user.

#### NCF

In order to run NCF: 
1) Run RML or R to load in the data
2) Run the corresponding PD-RML or PD-R depending on which data you loaded in 1)
3) Run TTV. This will create train, validation, and test splits.
4) Run BNCF. This will compile the NCF model.
5) Run TNCF. This will train the NCF model.
6) (Optional) Run ENCF. This will evaluate the model on the test split.
7) (Optional) Run SNCF. This will save the trained NCF model.
8) (Optional) Run RNCF. This will load a previously trained NCF model.
9) Run RECS, enter 'ncf' and answer the pop-up questions. This will return reccommendations for the given user.

## Evaluations

#### UU-CF

In order to evaluate the UU-CF model:
1) Run RML or R to load in the data
2) Run the corresponding PD-RML or PD-R depending on which data you loaded in 1)
3) Run SIMU.
4) Run LCVSIM, enter 'uu-cf' and the hyperparameters of the model.

#### II-CF

In order to evaluate the II-CF model:
1) Run RML or R to load in the data
2) Run the corresponding PD-RML or PD-R depending on which data you loaded in 1)
3) Run SIM.
4) Run LCVSIM, enter 'ii-cf' and the hyperparameters of the model.

#### MF-SGD

In order to evaluate the MF-SGD model: 
1) Run RML or R to load in the data
2) Run the corresponding PD-RML or PD-R depending on which data you loaded in 1)
3) Run T, enter Y. This will create train and test splits.
4) Run MF-SGD. Follow the pop-up questions and continue. Results will follow.

#### MF-SGD

In order to evaluate the MF-ALS model: 
1) Run RML or R to load in the data
2) Run the corresponding PD-RML or PD-R depending on which data you loaded in 1)
3) Run T, enter Y. This will create train and test splits.
4) Run MF-ALS. Follow the pop-up questions and continue. Results will follow.

#### TFIDF

In order to evaluate the TFIDF model:
1) Run RML or R to load in the data
2) Run the corresponding PD-RML or PD-R depending on which data you loaded in 1)
3) Run TFIDF.
4) Run LCVSIM, enter 'tfidf' and the hyperparameters of the model.

#### HYB

In order to evaluate the HYB model:
1) Run RML or R to load in the data
2) Run the corresponding PD-RML or PD-R depending on which data you loaded in 1)
3) Run TFIDF.
4) Run SIM.
5) Run HYB. This created the updated cosine similarity matrix.
6) Run LCVSIM, enter 'hybrid' and the hyperparameters of the model.

#### NCF

In order to run NCF: 
1) Run RML or R to load in the data
2) Run the corresponding PD-RML or PD-R depending on which data you loaded in 1)
3) Run TTV. This will create train, validation, and test splits.
4) Run BNCF or RNCF. This will compile the NCF model.
5) Run TNCF. This will train the NCF model.
6) Run ENCF. This will evaluate the model on the test split.


## Grid Searches

#### TFIDF

In order to perform the TFIDF grid search:
1) Run RML or R to load in the data
2) Run the corresponding PD-RML or PD-R depending on which data you loaded in 1)
3) Run TFIDF-GRID.

#### HYB

In order to perform the HYB grid search:
1) Run RML or R to load in the data
2) Run the corresponding PD-RML or PD-R depending on which data you loaded in 1)
3) Run HYB-GRID.

#### NCF

In order to perform the HYB grid search:
1) Run RML or R to load in the data
2) Run the corresponding PD-RML or PD-R depending on which data you loaded in 1)
3) Run NCF-GRID.

## Results

Results are saved inside Folder 2, but the location depends on the command. NCF model results are saved to Folder 2/ncf_models. 
LOOCV results are saved to the final_result_full.csv file. Our past results have been renamed and stored in individual folders, 
for example, Folder 2/results_hybrid. 

## Analysis

In Folder 2/analysis.ipynb, you can see the code used to analyze the results and produce graphs.





