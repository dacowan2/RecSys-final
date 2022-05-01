# RecSys-final

This repository contains files relating to our final project for CSC 381 (Recommender Systems) at Davidson College. The recommender system is contained in recommendations.py. The file contains implementations for of 10 different algorithms: 1) user-based collaborative filtering using Euclidean distance for user-user similarity calculations (UU-CF-D), 2) user-based collaborative filtering using Pearson similarity (UU-CF-P), 3) item-based collaborative filtering using Euclidean distance (II-CF-D), 4) item-based collaborative filtering using Pearson similarity (II-CF-P), 5) matrix factorization collaborative filtering with stochastic gradient descent (MF-SGD), 6) matrix factorization collaborative filtering with alternating least squares (MF-ALS), 7) TFIDF content-based method with cosine similarity for content similarity calculations (TFIDF), 8) a hybrid model using a combination of the II-CF-D and TFIDF methods, (HYB-D) 9) a hybrid model using a combination of the II-CF-P and TFIDF methods (HYB-P) 10) and a neural collaborative filtering model (NCF). Intructions for running each of these models are below:


## UU-CF

In order to run UU-CF: 
1) Run RML or R to load in the data
2) Run the corresponding PD-RML or PD-R depending on which data you loaded in 1)
3) Run SIMU, read or write the user-user similarity matrix with a specificied similarity metric.
4) Run RECS, enter 'uu-cf' and answer the pop-up questions. When you are asked for the similarity calculation, enter 'd' for distance or 'p' for pearson. 

## II-CF

In order to run II-CF: 
1) Run RML or R to load in the data
2) Run the corresponding PD-RML or PD-R depending on which data you loaded in 1)
3) Run SIM, read or write the item-item similarity matrix with a specificied similarity metric.
4) Run RECS, enter 'ii-cf' and answer the pop-up questions. When you are asked for the similarity calculation, enter 'd' for distance or 'p' for pearson.

## MF-SGD

In order to run MF-SGD: 
1) Run RML or R to load in the data
2) Run the corresponding PD-RML or PD-R depending on which data you loaded in 1)
3) Run T, enter Y. This will create train and test splits.
4) Run MF-SGD. Follow the pop-up questions and continue.
5) Run RECS, enter 'mf-sgd' and answer the pop-up questions. This will return reccommendations for the given user.

## MF-ALS

In order to run MF-ALS: 
1) Run RML or R to load in the data
2) Run the corresponding PD-RML or PD-R depending on which data you loaded in 1)
3) Run T, enter Y. This will create train and test splits.
4) Run MF-ALS. Follow the pop-up questions and continue.
5) Run RECS, enter 'mf-als' and answer the pop-up questions. This will return reccommendations for the given user.


