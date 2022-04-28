"""
CSC381: Building a simple Recommender System
The final code package is a collaborative programming effort between the
CSC381 student(s) named below, the class instructor (Carlos Seminario), and
source code from Programming Collective Intelligence, Segaran 2007.
This code is for academic use/purposes only.
CSC381 Programmer/Researcher: Brad Shook, Daniel Cowan, Drew Dibble
"""

import os
from math import sqrt
from numpy import mean, std, array, median, square
from scipy.stats import spearmanr
from scipy.stats import kendalltau
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import time
import warnings
import traceback
from turtle import color
import numpy as np
import math
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import csv
from os import getcwd
from numpy.linalg import solve ## needed for als
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from time import time
from copy import deepcopy
import sklearn.model_selection
from keras.models import load_model
from keras.callbacks import EarlyStopping
import sklearn.metrics
from keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate, BatchNormalization, Dropout
from keras.models import Model, load_model
from keras.losses import MeanSquaredError
from keras.optimizers import Adam

warnings.filterwarnings("ignore")  # ("once") #("module") #("default") #("error")

MAX_PRINT = 10


def from_file_to_dict(path, datafile, itemfile):
    """Load user-item matrix from specified file

    Parameters:
    -- path: directory path to datafile and itemfile
    -- datafile: delimited file containing userid, itemid, rating
    -- itemfile: delimited file that maps itemid to item name

    Returns:
    -- prefs: a nested dictionary containing item ratings for each user

    """

    # Get movie titles, place into movies dictionary indexed by itemID
    movies = {}
    try:
        with open(path + "/" + itemfile, encoding="iso8859") as myfile:
            # this encoding is required for some datasets: encoding='iso8859'
            for line in myfile:
                (id, title) = line.split("|")[0:2]
                movies[id] = title.strip()

    # Error processing
    except UnicodeDecodeError as ex:
        print(ex)
        print(len(movies), line, id, title)
        return {}
    except Exception as ex:
        print(ex)
        print(len(movies))
        return {}

    # Load data into a nested dictionary
    prefs = {}
    for line in open(path + "/" + datafile, encoding="iso8859"):
        # print(line, line.split('\t')) #debug
        (user, movieid, rating, ts) = line.split("\t")
        user = user.strip()  # remove spaces
        movieid = movieid.strip()  # remove spaces
        prefs.setdefault(user, {})  # make it a nested dicitonary
        prefs[user][movies[movieid]] = float(rating)

    # return a dictionary of preferences
    return prefs


def data_stats(prefs, filename):
    """Computes/prints descriptive analytics:
    -- Total number of users, items, ratings
    -- Overall average rating, standard dev (all users, all items)
    -- Average item rating, standard dev (all users)
    -- Average user rating, standard dev (all items)
    -- Matrix ratings sparsity
    -- Ratings distribution histogram (all users, all items)
    Parameters:
    -- prefs: dictionary containing user-item matrix
    -- filename: string containing name of file being analyzed
    Returns:
    -- None
    """

    users = prefs.keys()

    # Loop through each user's movie ratings
    ratings_list = []
    movie_titles_list = []
    ratings_per_user_dict = {}
    for ratings_per_user in prefs.values():
        ratings = ratings_per_user.values()
        movies = ratings_per_user.keys()

        # Create a list of all the ratings in the UI matrix
        for rating in ratings:
            ratings_list.append(rating)

        # Get a list of movie titles
        for movie in movies:
            movie_titles_list.append(movie)

    # Convert list to set to remove duplicates
    movie_titles_set = set(movie_titles_list)

    # Make a dictionary with keys as users and values as [sum of ratings, number of ratings]
    for user in users:
        ratings = prefs[user].values()
        for rating in ratings:
            if user not in ratings_per_user_dict.keys():
                ratings_per_user_dict[user] = [rating, 1]
            else:
                ratings_per_user_dict[user][0] += rating
                ratings_per_user_dict[user][1] += 1

    # Make a dictionary with keys as movies and values as [sum of ratings, number of ratings]
    ratings_per_movie_dict = {}
    for user in prefs.values():
        for movie in movie_titles_set:
            if (movie not in ratings_per_movie_dict.keys()) and (movie in user.keys()):
                ratings_per_movie_dict[movie] = [user[movie], 1]
            elif (movie in ratings_per_movie_dict.keys()) and (movie in user.keys()):
                ratings_per_movie_dict[movie][0] += user[movie]
                ratings_per_movie_dict[movie][1] += 1

    # Calculate the average rating per movie
    avg_rating_per_movie_list = []
    for movie in movie_titles_set:
        avg_rating_per_movie_list.append(
            ratings_per_movie_dict[movie][0] / ratings_per_movie_dict[movie][1]
        )

    # Calculate the average rating per user
    avg_rating_per_user_list = []
    for user in users:
        avg_rating_per_user_list.append(
            ratings_per_user_dict[user][0] / ratings_per_user_dict[user][1]
        )

    overall_average_rating = mean(ratings_list)
    overall_rating_std = std(ratings_list)
    average_item_rating = mean(avg_rating_per_movie_list)
    average_item_rating_std = std(avg_rating_per_movie_list)
    average_user_rating = mean(avg_rating_per_user_list)
    average_user_rating_std = std(avg_rating_per_user_list)

    n_users = len(prefs.keys())
    n_ratings = len(ratings_list)

    # Find number of items
    n_items = len(movie_titles_set)

    # Find average number of ratings per user
    ratings_per_user_list = [x[1] for x in list(ratings_per_user_dict.values())]
    avg_ratings_per_user = mean(ratings_per_user_list)
    std_ratings_per_user = std(ratings_per_user_list)
    min_ratings_per_user = min(ratings_per_user_list)
    max_ratings_per_user = max(ratings_per_user_list)
    median_ratings_per_user = median(ratings_per_user_list)

    # Print stats
    print("Stats for: {}\n".format(filename))
    print("Number of users: {}".format(n_users))
    print("Number of items: {}".format(n_items))
    print("Number of ratings: {}".format(n_ratings))
    print(
        "Overall average rating: {overall_mean:.2f} out of 5, and std dev of {overall_std:.2f}".format(
            overall_mean=overall_average_rating, overall_std=overall_rating_std
        )
    )

    if len(avg_rating_per_movie_list) < 50:
        print("Item Avg Ratings List: " + str(avg_rating_per_movie_list))

    print(
        "Average item rating: {avg_rating_per_movie:.2f} out of 5, and std dev of {std_rating_per_movie:.2f}".format(
            avg_rating_per_movie=average_item_rating,
            std_rating_per_movie=average_item_rating_std,
        )
    )
    if len(avg_rating_per_user_list) < 50:
        print("User Ratings List: " + str(avg_rating_per_user_list))

    print(
        "Average user rating: {avg_rating_per_user:.2f} out of 5, and std dev of {std_rating_per_user:.2f}".format(
            avg_rating_per_user=average_user_rating,
            std_rating_per_user=average_user_rating_std,
        )
    )

    print(
        f"Average number of ratings per user: {avg_ratings_per_user:.2f} with std dev of {std_ratings_per_user:.2f}"
    )
    print(f"Min number of ratings for single user: {min_ratings_per_user}")
    print(f"Max number of ratings for single user: {max_ratings_per_user}")
    print(f"Median number of ratings for single user: {int(median_ratings_per_user)}")

    print(
        "User-Item Matrix Sparsity: {sparsity:.2f}%".format(
            sparsity=100 * (1 - (n_ratings) / (n_users * n_items))
        )
    )
    print()
    # Plot ratings histrogram
    plt.hist(ratings_list, bins=4, range=(1, 5), color="#ac1a2f")
    plt.ylabel("number of user ratings")
    plt.xticks([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    plt.xlabel("rating")
    plt.xlim((1, 5))
    plt.title("Ratings Distribution for ML100k")
    plt.savefig("ratings_hist.png")
    plt.show()


def popular_items(prefs, filename):
    """Computes/prints popular items analytics
    -- popular items: most rated (sorted by # ratings)
    -- popular items: highest rated (sorted by avg rating)
    -- popular items: highest rated items that have at least a
                      "threshold" number of ratings
    Parameters:
    -- prefs: dictionary containing user-item matrix
    -- filename: string containing name of file being analyzed
    Returns:
    -- None
    """
    df = pd.DataFrame(prefs).T

    num_ratings_item = dict(df.count())
    num_ratings_item_sort = dict(
        sorted(num_ratings_item.items(), key=lambda x: x[1], reverse=True)
    )

    avg_rating_item = dict(df.mean())
    avg_rating_item_sort = dict(
        sorted(avg_rating_item.items(), key=lambda x: x[1], reverse=True)
    )

    threshold = 100

    avg_item_thresh = df.mean().where(df.count() >= threshold).dropna()
    avg_item_thresh_sort = dict(
        sorted(avg_item_thresh.items(), key=lambda x: x[1], reverse=True)
    )

    digits = max([len(x) for x in list(num_ratings_item_sort.keys())[:15]])
    offset = 5

    print("MOST RATED:\n")
    print(f"{'Name' : <{digits+offset}}{'Number of Ratings'}")
    for i, item in enumerate(num_ratings_item_sort):
        if i >= 15:
            break
        rating = num_ratings_item_sort[item]
        print(f"{item : <{digits+offset}}{rating}")

    digits = max([len(x) for x in list(avg_rating_item_sort.keys())[:15]])

    print("\nHIGHEST RATED:\n")
    print(f"{'Name' : <{digits+offset}}{'Average Rating'}")
    for i, item in enumerate(avg_rating_item_sort):
        if i >= 15:
            break
        rating = avg_rating_item_sort[item]
        print(f"{item : <{digits+offset}}{rating:.3f}")

    digits = max([len(x) for x in list(avg_item_thresh_sort.keys())[:15]])

    print(f"\nHIGHEST RATED w/ AT LEAST {threshold} RATINGS:\n")
    print(
        f"{'Name' : <{digits+offset}}{'Average Rating': <{20}}{'Number of Times Rated'}"
    )
    for i, item in enumerate(avg_item_thresh_sort):
        if i >= 15:
            break
        rating = avg_item_thresh_sort[item]
        num_ratings = num_ratings_item[item]
        print(f"{item : <{digits+offset}}{rating: <{20}.3f}{num_ratings}")

    return


# Returns a distance-based similarity score for person1 and person2
def sim_distance(prefs, person1, person2, sim_sig_weighting):
    """
        Calculate Euclidean distance similarity

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person1: string containing name of user 1
        -- person2: string containing name of user 2

        Returns:
        -- Euclidean distance similarity as a float

    Source: Programming Collective Intelligence, Segaran 2007
    """

    # Get the list of shared_items
    si = {}
    for item in prefs[person1]:
        if item in prefs[person2]:
            si[item] = 1

    # if they have no ratings in common, return 0
    if len(si) == 0:
        return 0

    # Add up the squares of all the differences
    sum_of_squares = sum(
        [
            pow(prefs[person1][item] - prefs[person2][item], 2)
            for item in prefs[person1]
            if item in prefs[person2]
        ]
    )

    similarity = 1 / (1 + sqrt(sum_of_squares))

    if len(si) < sim_sig_weighting and sim_sig_weighting != 1:
        return similarity * (len(si) / sim_sig_weighting)

    """
    ## FYI, This is what the list comprehension above breaks down to ..
    ##
    sum_of_squares = 0
    for item in prefs[person1]:
        if item in prefs[person2]:
            #print(item, prefs[person1][item], prefs[person2][item])
            sq = pow(prefs[person1][item]-prefs[person2][item],2)
            #print (sq)
            sum_of_squares += sq
    """

    return similarity


# Returns the Pearson correlation coefficient for p1 and p2
def sim_pearson(prefs, p1, p2, sim_sig_weighting):
    """
        Calculate Pearson Correlation similarity

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person1: string containing name of user 1
        -- person2: string containing name of user 2

        Returns:
        -- Pearson Correlation similarity as a float

    Source: Programming Collective Intelligence, Segaran 2007
    """
    # Get the list of mutually rated items
    si = {}
    for item in prefs[p1]:
        if item in prefs[p2]:
            si[item] = 1

    # if there are no ratings in common, return 0
    if len(si) == 0:
        return 0

    # Sum calculations
    n = len(si)

    # Sums of all the preferences
    sum1 = sum([prefs[p1][it] for it in si])
    sum2 = sum([prefs[p2][it] for it in si])

    # Sums of the squares
    sum1Sq = sum([pow(prefs[p1][it], 2) for it in si])
    sum2Sq = sum([pow(prefs[p2][it], 2) for it in si])

    # Sum of the products
    pSum = sum([prefs[p1][it] * prefs[p2][it] for it in si])

    # Calculate r (Pearson score)
    num = pSum - (sum1 * sum2 / n)
    den = sqrt((sum1Sq - pow(sum1, 2) / n) * (sum2Sq - pow(sum2, 2) / n))
    if den == 0:
        return 0

    r = num / den

    if len(si) < sim_sig_weighting and sim_sig_weighting != 1:
        return r * (len(si) / sim_sig_weighting)

    return r


def sim_tanimoto(prefs, p1, p2, sim_sig_weighting):
    """
    Returns the Tanimoto correlation coefficient for vectors p1 and p2
    https://en.wikipedia.org/wiki/Jaccard_index#Tanimoto_similarity_and_distance

    """

    # Get the list of mutually rated items
    si = {}
    for item in prefs[p1]:
        if item in prefs[p2]:
            si[item] = 1

    # if there are no ratings in common, return 0
    if len(si) == 0:
        return 0

    # Sum calculations
    n = len(si)

    # Sums of the squares
    sum1Sq = sum([pow(prefs[p1][it], 2) for it in si])
    sum2Sq = sum([pow(prefs[p2][it], 2) for it in si])

    # Sum of the products
    pSum = sum([prefs[p1][it] * prefs[p2][it] for it in si])

    # Calculate r (Tanimoto score)
    num = pSum
    den = sum1Sq + sum2Sq - pSum
    if den == 0:
        return 0

    r = num / den

    if len(si) < sim_sig_weighting and sim_sig_weighting != 1:
        return r * (len(si) / sim_sig_weighting)

    return r


def sim_jaccard(prefs, p1, p2, sim_sig_weighting):

    """
    The Jaccard similarity index (sometimes called the Jaccard similarity coefficient)
    compares members for two sets to see which members are shared and which are distinct.
    It’s a measure of similarity for the two sets of data, with a range from 0% to 100%.
    The higher the percentage, the more similar the two populations. Although it’s easy to
    interpret, it is extremely sensitive to small samples sizes and may give erroneous
    results, especially with very small samples or data sets with missing observations.
    https://www.statisticshowto.datasciencecentral.com/jaccard-index/
    https://en.wikipedia.org/wiki/Jaccard_index

    The formula to find the Index is:
    Jaccard Index = (the number in both sets) / (the number in either set) * 100

    In Steps, that’s:
    Count the number of members which are shared between both sets.
    Count the total number of members in both sets (shared and un-shared).
    Divide the number of shared members (1) by the total number of members (2).
    Multiply the number you found in (3) by 100.

    A simple example using set notation: How similar are these two sets?

    A = {0,1,2,5,6}
    B = {0,2,3,4,5,7,9}

    Solution: J(A,B) = |A∩B| / |A∪B| = |{0,2,5}| / |{0,1,2,3,4,5,6,7,9}| = 3/9 = 0.33.

    Notes:
    The cardinality of A, denoted |A| is a count of the number of elements in set A.
    Although it’s customary to leave the answer in decimal form if you’re using set
    notation, you could multiply by 100 to get a similarity of 33.33%.

    """
    # Get the lists of mutually rated and unique items
    common_items = {}
    unique_items = {}
    for item in prefs[p1]:
        if item in prefs[p2]:
            # common_items[item]=1 # Case 0: count as common_items if item is rated in both lists
            if prefs[p1][item] == prefs[p2][item]:  # Case 1: rating must match exactly!
                # if abs(prefs[p1][item] - prefs[p2][item]) <= 0.5: # Case 2: rating must be +/- 0.5!
                common_items[item] = 1
            else:
                unique_items[item] = 1
        else:
            unique_items[item] = 1

    # if there are no ratings in common, return 0
    if len(common_items) == 0:
        return 0

    # Sum calculations
    num = len(common_items)

    # Calculate Jaccard index
    den = len(common_items) + len(unique_items)
    if den == 0:
        return 0

    jaccard_index = num / den

    if len(common_items) < sim_sig_weighting and sim_sig_weighting != 1:
        return jaccard_index * (len(common_items) / sim_sig_weighting)

    return jaccard_index


def sim_cosine(prefs, p1, p2, sim_sig_weighting):
    """
    Info:
    https://www.geeksforgeeks.org/cosine-similarity/
    https://en.wikipedia.org/wiki/Cosine_similarity

    Source for some of the code: https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists

    """
    # Get the list of mutually rated items
    si = {}
    for item in prefs[p1]:
        if item in prefs[p2]:
            si[item] = 1

    # if there are no ratings in common, return 0
    if len(si) == 0:
        return 0

    # Sum of the products
    sumxy = [prefs[p1][it] * prefs[p2][it] for it in si]
    sumxx = [prefs[p1][it] * prefs[p1][it] for it in si]
    sumyy = [prefs[p2][it] * prefs[p2][it] for it in si]
    # print (sumxy, sumxx, sumyy)
    sumxy = sum(sumxy)
    sumxx = sum(sumxx)
    sumyy = sum(sumyy)

    # Calculate r (cosine sim score)
    num = sumxy
    den = sqrt(sumxx * sumyy)

    if den == 0:
        return 0

    r = num / den

    if len(si) < sim_sig_weighting and sim_sig_weighting != 1:
        return r * (len(si) / sim_sig_weighting)

    return r


def sim_spearman(prefs, p1, p2, sim_sig_weighting):
    """
    Calc Spearman's correlation coefficient using scipy function

    Enter >>> help(spearmanr) # to get helpful info
    """

    # Get the list of mutually rated items
    si = {}
    for item in prefs[p1]:
        if item in prefs[p2]:
            si[item] = 1

    # if there are no ratings in common, return 0
    if len(si) == 0:
        return 0

    # Sum calculations
    n = len(si)

    # Sums of all the preferences
    data1 = [prefs[p1][it] for it in si]
    data2 = [prefs[p2][it] for it in si]

    len1 = len(data1)
    len2 = len(data2)

    coef, p = spearmanr(data1, data2)
    # print('Spearmans correlation coefficient: %.3f' % coef)

    if str(coef) == "nan":
        return 0

    # interpret the significance
    """
    alpha = 0.05
    if p > alpha:
        print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
    else:
        print('Samples are correlated (reject H0) p=%.3f' % p)   
    
    """

    if len(si) < sim_sig_weighting and sim_sig_weighting != 1:
        return coef * (len(si) / sim_sig_weighting)

    return coef


def sim_kendall_tau(prefs, p1, p2, sim_sig_weighting):
    """
    Calc Kendall Tau correlation coefficient using scipy function

    Enter >>> help(kendalltau) # to get helpful info
    """

    # Get the list of mutually rated items
    si = {}
    for item in prefs[p1]:
        if item in prefs[p2]:
            si[item] = 1

    # if there are no ratings in common, return 0
    if len(si) == 0:
        return 0

    # Sum calculations
    n = len(si)

    # Sums of all the preferences
    data1 = [prefs[p1][it] for it in si]
    data2 = [prefs[p2][it] for it in si]

    len1 = len(data1)
    len2 = len(data2)

    coef, p = kendalltau(data1, data2)

    if -1 <= coef <= 1:
        pass
    else:
        coef = 0
        # print(coef, p1, p2)

    # sum_coef = 0
    # for it in si:
    # coef, p = kendalltau(prefs[p1][it], prefs[p2][it])
    # coef, p = kendalltau(p1, p2)
    # sum_coef += coef
    # print('Kendall correlation coefficient: %.3f' % coef)
    # coef = sum_coef/n

    """
    # interpret the significance
    alpha = 0.05
    if p > alpha:
        print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
    else:
        print('Samples are correlated (reject H0) p=%.3f' % p)    
    """

    if len(si) < sim_sig_weighting and sim_sig_weighting != 1:
        return coef * (len(si) / sim_sig_weighting)

    return coef


def getRecommendations(prefs, person, similarity=sim_pearson):
    """
    Calculates recommendations for a given user
    Parameters:
    -- prefs: dictionary containing user-item matrix
    -- person: string containing name of user
    -- similarity: function to calc similarity (sim_pearson is default)
    Returns:
    -- A list of recommended items with 0 or more tuples,
       each tuple contains (predicted rating, item name).
       List is sorted, high to low, by predicted rating.
       An empty list is returned when no recommendations have been calc'd.
    """

    totals = {}
    simSums = {}
    for other in prefs:
        # don't compare me to myself
        if other == person:
            continue
        sim = similarity(prefs, person, other)

        # ignore scores of zero or lower
        if sim <= 0:
            continue
        for item in prefs[other]:

            # only score movies I haven't seen yet
            if item not in prefs[person] or prefs[person][item] == 0:
                # Similarity * Score
                totals.setdefault(item, 0)
                totals[item] += prefs[other][item] * sim
                # Sum of similarities
                simSums.setdefault(item, 0)
                simSums[item] += sim

    # Create the normalized list
    rankings = [(total / simSums[item], item) for item, total in totals.items()]

    # Return the sorted list
    rankings.sort()
    rankings.reverse()
    return rankings


def get_all_UU_recs(prefs, sim=sim_pearson, num_users=10, top_N=5):
    """
    Print user-based CF recommendations for all users in dataset
    Parameters
    -- prefs: nested dictionary containing a U-I matrix
    -- sim: similarity function to use (default = sim_pearson)
    -- num_users: max number of users to print (default = 10)
    -- top_N: max number of recommendations to print per user (default = 5)
    Returns: None
    """

    users = list(prefs.keys())

    print("Using sim_pearson: ")
    for count, user in enumerate(users):
        if count > num_users:
            break
        recs = getRecommendations(prefs, user, similarity=sim_pearson)[:top_N]
        print("User-based CF recs for " + user + ": " + str(recs))

    print()

    print("Using sim_distance: ")
    for count, user in enumerate(users):
        if count > num_users:
            break
        recs = getRecommendations(prefs, user, similarity=sim_distance)[:top_N]
        print("User-based CF recs for " + user + ": " + str(recs))


def loo_cv(prefs, metric, sim, algo, dataset_name):
    """
    Leave_One_Out Evaluation: evaluates recommender system ACCURACY
     Parameters:
         prefs dataset: critics, ml-100K, etc.
         metric: MSE, MAE, RMSE, etc.
         sim: distance, pearson, etc.
         algo: user-based recommender, item-based recommender, etc.
    Returns:
         error_total: MSE, MAE, RMSE totals for this set of conditions
         error_list: list of actual-predicted differences
    """

    users = list(prefs.keys())
    prefs_df = pd.DataFrame.from_dict(prefs, orient="index")
    items = list(prefs_df.columns)
    error_list = []

    start = time.time()
    for i, user in enumerate(users):
        for item in items:
            # if user has rated item
            if item in prefs[user].keys():
                removed_rating = prefs[user][item]

                del prefs[user][item]

                recs = algo(prefs, user, similarity=sim)

                prediction = [rec for rec in recs if item in rec]

                if prediction != []:
                    curr_error = prediction[0][0] - removed_rating
                    error_list.append(curr_error)

                prefs[user][item] = removed_rating

        if i % 10 == 0 and not i == 0:
            secs = time.time() - start
            print(f"Number of users processed: {i}")
            print(f"==>> {secs} secs for {i} items, secs per item {secs/i}")

            error_array = array(error_list)

            mse = (1 / len(error_list)) * sum(error_array ** 2)
            rmse = sqrt((1 / len(error_list)) * sum(error_array ** 2))
            mae = (1 / len(error_list)) * sum(abs(error_array))
            print(f"MSE: {round(mse,5)}, RMSE: {round(rmse,5)}, MAE: {round(mae,5)}")

    print(
        "Final results froom loo_cv_sim: len(prefs)={prefs_length_val}, sim={sim_val}, {function_val}".format(
            prefs_length_val=len(prefs), sim_val=sim, function_val=algo
        )
    )

    print(f"MSE: {round(mse,5)}, RMSE: {round(rmse,5)}, MAE: {round(mae,5)}")
    print(f"MSE for {dataset_name}, len(SE list): len(")

    return (mse, rmse, mae), error_list


def topMatches(prefs, person, similarity=sim_pearson, n=100, sim_sig_weighting=1):
    """
    Returns the best matches for person from the prefs dictionary
    Parameters:
    -- prefs: dictionary containing user-item matrix
    -- person: string containing name of user
    -- similarity: function to calc similarity (sim_pearson is default)
    -- n: number of matches to find/return (5 is default)
    Returns:
    -- A list of similar matches with 0 or more tuples,
       each tuple contains (similarity, item name).
       List is sorted, high to low, by similarity.
       An empty list is returned when no matches have been calc'd.
    """
    scores = [
        (similarity(prefs, person, other, sim_sig_weighting), other)
        for other in prefs
        if other != person
    ]
    scores.sort()
    scores.reverse()
    return scores[0:n]


def transformPrefs(prefs):
    """
    Transposes U-I matrix (prefs dictionary)
    Parameters:
    -- prefs: dictionary containing user-item matrix
    Returns:
    -- A transposed U-I matrix, i.e., if prefs was a U-I matrix,
       this function returns an I-U matrix
    """
    result = {}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item, {})
            # Flip item and person
            result[item][person] = prefs[person][item]
    return result


def calculateSimilarItems(
    prefs, neighbors=100, similarity=sim_pearson, sim_sig_weighting=1
):
    """
    Creates a dictionary of items showing which other items they are most
    similar to.
    Parameters:
    -- prefs: dictionary containing user-item matrix
    -- n: number of similar matches for topMatches() to return
    -- similarity: function to calc similarity (sim_pearson is default)
    Returns:
    -- A dictionary with a similarity matrix
    """
    result = {}
    # Invert the preference matrix to be item-centric
    itemPrefs = transformPrefs(prefs)

    start = time.time()

    for i, item in enumerate(itemPrefs):
        # Status updates for larger datasets
        if i % 100 == 0 and not i == 0:
            secs = time.time() - start
            print(f"Processed {i} out of {len(itemPrefs)} items")
            print(f"==>> {secs} secs for {i} items, secs per item {secs/i}")

        # Find the most similar items to this one
        scores = topMatches(
            itemPrefs,
            item,
            similarity,
            n=neighbors,
            sim_sig_weighting=sim_sig_weighting,
        )
        result[item] = scores
    return result


def calculateSimilarUsers(
    prefs, neighbors=100, similarity=sim_pearson, sim_sig_weighting=1
):
    """
    Creates a dictionary of users showing which other users they are most
    similar to.
    Parameters:
    -- prefs: dictionary containing user-item matrix
    -- n: number of similar matches for topMatches() to return
    -- similarity: function to calc similarity (sim_pearson is default)
    Returns:
    -- A dictionary with a similarity matrix
    """
    result = {}
    start = time.time()
    for i, user in enumerate(prefs):
        # Status updates for larger datasets
        if i % 100 == 0 and not i == 0:
            secs = time.time() - start
            print(f"Processed {i} out of {len(prefs)} items")
            print(f"==>> {secs} secs for {i} items, secs per item {secs/i}")

        # Find the most similar items to this one
        scores = topMatches(
            prefs, user, similarity, n=neighbors, sim_sig_weighting=sim_sig_weighting
        )
        result[user] = scores

    return result


def getRecommendedItems(prefs, itemMatch, user, threshold):
    """
    Calculates recommendations for a given user
    Parameters:
    -- prefs: dictionary containing user-item matrix
    -- itemMatch: item-item similarity matrix
    -- person: string containing name of user
    Returns:
    -- A list of recommended items with 0 or more tuples,
       each tuple contains (predicted rating, item name).
       List is sorted, high to low, by predicted rating.
       An empty list is returned when no recommendations have been calc'd.
    """
    userRatings = prefs[user]
    scores = {}
    totalSim = {}
    # Loop over items rated by this user
    for (item, rating) in userRatings.items():

        # Loop over items similar to this one
        for (similarity, item2) in itemMatch[item]:

            # Ignore if this user has already rated this item
            if item2 in userRatings:
                continue
            # ignore scores of zero or lower
            if similarity <= threshold:
                continue
            # Weighted sum of rating times similarity
            scores.setdefault(item2, 0)
            scores[item2] += similarity * rating
            # Sum of all the similarities
            totalSim.setdefault(item2, 0)
            totalSim[item2] += similarity

    # Divide each total score by total weighting to get an average
    print('scores items')
    print(scores.items())
    rankings = [(score / totalSim[item], item) for item, score in scores.items()]

    # Return the rankings from highest to lowest
    rankings.sort()
    rankings.reverse()
    return rankings


def getRecommendedUsers(prefs, userMatch, user, threshold):
    """
    Calculates recommendations for a given user
    Parameters:
    -- prefs: dictionary containing user-item matrix
    -- itemMatch: item-item similarity matrix
    -- person: string containing name of user
    Returns:
    -- A list of recommended items with 0 or more tuples,
       each tuple contains (predicted rating, item name).
       List is sorted, high to low, by predicted rating.
       An empty list is returned when no recommendations have been calc'd.
    """
    totals = {}
    simSums = {}
    for other in prefs:
        # don't compare me to myself
        if other == user:
            continue

        sim = 0
        for row in userMatch[user]:
            if row[1] == other:
                sim = row[0]

        # ignore scores of zero or lower
        if sim <= threshold:
            continue

        for item in prefs[other]:
            # only score movies I haven't seen yet
            if item not in prefs[user] or prefs[user][item] == 0:
                # Similarity * Score
                totals.setdefault(item, 0)
                totals[item] += prefs[other][item] * sim
                # Sum of similarities
                simSums.setdefault(item, 0)
                simSums[item] += sim

    # Create the normalized list
    rankings = [(total / simSums[item], item) for item, total in totals.items()]

    # Return the sorted list
    rankings.sort()
    rankings.reverse()
    return rankings


def new_getRecommendedItems(prefs, itemMatch, user, threshold, cur_item, movies):
    """
    Calculates recommendations for a given user
    Parameters:
    -- prefs: dictionary containing user-item matrix
    -- itemMatch: item-item similarity matrix
    -- person: string containing name of user
    Returns:
    -- A list of recommended items with 0 or more tuples,
       each tuple contains (predicted rating, item name).
       List is sorted, high to low, by predicted rating.
       An empty list is returned when no recommendations have been calc'd.
    """
    userRatings = prefs[user]
    scores = {}
    totalSim = {}

    # Loop over items rated by this user
    for (item, rating) in userRatings.items():
        # Loop over items similar to this one
        for (similarity, item2) in itemMatch[item]:
            if not item2 == cur_item:
                continue

            # Ignore if this user has already rated this item
            if item2 in userRatings:
                continue
            # ignore scores of zero or lower
            if similarity <= threshold:
                continue
            # Weighted sum of rating times similarity
            scores.setdefault(item2, 0)
            scores[item2] += similarity * rating
            # Sum of all the similarities
            totalSim.setdefault(item2, 0)
            totalSim[item2] += similarity

    # Divide each total score by total weighting to get an average

    rankings = [
        (score / totalSim[item], item)
        for item, score in scores.items()
        if item == cur_item
    ]

    # Return the rankings from highest to lowest
    rankings.sort()
    rankings.reverse()
    return rankings


def new_getRecommendedUsers(prefs, userMatch, user, threshold, cur_item, movies):
    """
    Calculates recommendations for a given user
    Parameters:
    -- prefs: dictionary containing user-item matrix
    -- itemMatch: item-item similarity matrix
    -- person: string containing name of user
    Returns:
    -- A list of recommended items with 0 or more tuples,
       each tuple contains (predicted rating, item name).
       List is sorted, high to low, by predicted rating.
       An empty list is returned when no recommendations have been calc'd.
    """
    totals = {}
    simSums = {}
    for other in prefs:
        # don't compare me to myself
        if other == user:
            continue

        sim = 0
        for row in userMatch[user]:
            if row[1] == other:
                sim = row[0]
                break

        # ignore scores of zero or lower
        if sim <= threshold:
            continue

        for item in prefs[other]:
            if not item == cur_item:
                continue
            # only score movies I haven't seen yet
            if item not in prefs[user] or prefs[user][item] == 0:
                # Similarity * Score
                totals.setdefault(item, 0)
                totals[item] += prefs[other][item] * sim
                # Sum of similarities
                simSums.setdefault(item, 0)
                simSums[item] += sim

    print('Totals items: ')
    print(totals.items())
    # Create the normalized list
    rankings = [
        (total / simSums[item], item)
        for item, total in totals.items()
        if item == cur_item
    ]

    # Return the sorted list
    rankings.sort()
    rankings.reverse()
    return rankings


def get_all_II_recs(prefs, itemsim, sim_method, num_users=10, top_N=5):
    """
    Print item-based CF recommendations for all users in dataset
    Parameters
    -- prefs: U-I matrix (nested dictionary)
    -- itemsim: item-item similarity matrix (nested dictionary)
    -- sim_method: name of similarity method used to calc sim matrix (string)
    -- num_users: max number of users to print (integer, default = 10)
    -- top_N: max number of recommendations to print per user (integer, default = 5)
    Returns: None
    """
    users = list(prefs.keys())
    for user in users[:num_users]:
        recs = getRecommendedItems(prefs, itemsim, user)[:top_N]
        print(
            "Item-based CF recs for {user_value}, {sim_method_value}: {recs_val}".format(
                user_value=user, sim_method_value=sim_method, recs_val=recs
            )
        )


def loo_cv_sim(
    prefs, sim, algo, sim_matrix, dataset_name, threshold, sim_sig_weighting, neighbors, movies, weighting_factor = 0
):
    """
    Leave-One_Out Evaluation: evaluates recommender system ACCURACY
     Parameters:
         prefs dataset: critics, etc.
         metric: MSE, or MAE, or RMSE
         sim: distance, pearson, etc.
         algo: user-based recommender, item-based recommender, etc.
         sim_matrix: pre-computed similarity matrix
    Returns:
         error_total: MSE, or MAE, or RMSE totals for this set of conditions
         error_list: list of actual-predicted differences
    """

    print('Sim matrix')
    print(sim_matrix)

    users = list(prefs.keys())
    error_list = []

    start = time.time()
    for i, user in enumerate(users):

        if i % 10 == 0 and not i == 0:
            secs = time.time() - start
            print(f"Number of users processed: {i}")
            print(f"==>> {secs} secs for {i} users, secs per user {secs/i}")

            error_array = array(error_list)

            if len(error_list) > 0:
                mse = (1 / len(error_list)) * sum(error_array ** 2)
                rmse = sqrt((1 / len(error_list)) * sum(error_array ** 2))
                mae = (1 / len(error_list)) * sum(abs(error_array))
                print(
                    f"MSE: {round(mse,5)}, RMSE: {round(rmse,5)}, MAE: {round(mae,5)}"
                )
            else:
                print("No errors yet!")

        for item in list(prefs[user].keys()):
            removed_rating = prefs[user][item]
            # print(f'removed rating: {removed_rating}')
            # print(f'User: {user}, Item: {item}')

            del prefs[user][item]


            # Get a list of predicted ratings for item
            prediction = algo(prefs, sim_matrix, user, threshold, item, movies)

            if prediction != []:
                # Calc error between highest predicted rating and removed rating
                curr_error = prediction[0][0] - removed_rating
                error_list.append(curr_error)

            prefs[user][item] = removed_rating
        

    secs = time.time() - start
    print(f"Number of users processed: {i}")
    print(f"==>> {secs} secs for {i} users, secs per user {secs/i}")

    error_array = array(error_list)

    if error_array != []:

        mse = (1 / len(error_list)) * sum(error_array ** 2)
        rmse = sqrt((1 / len(error_list)) * sum(error_array ** 2))
        mae = (1 / len(error_list)) * sum(abs(error_array))
        print(f"MSE: {round(mse,5)}, RMSE: {round(rmse,5)}, MAE: {round(mae,5)}")
        print(
            "Final results froom loo_cv_sim: len(prefs)={prefs_length_val}, sim={sim_val}, {function_val}".format(
                prefs_length_val=len(prefs), sim_val=sim, function_val=algo
            )
        )

        print(f"MSE: {round(mse,5)}, RMSE: {round(rmse,5)}, MAE: {round(mae,5)}")
        print(
            f"MSE for {dataset_name}: {round(mse,5)}, len(SE list): {len(error_array)} using {sim}"
        )

        coverage = len(error_array) / sum([len(prefs[person].values()) for person in prefs])

    else:
        mse = -1
        rmse = -1
        mae = -1
        coverage = -1

    sim_str = str(sim)
    algo_str = str(algo).split()[1]

    cur_res_dict = {
        "dataset_name": [dataset_name],
        "sim_method": [sim_str],
        "algo": [algo_str],
        "sim_threshold": [threshold],
        "neighbors": [neighbors],
        "sig_weight": [sim_sig_weighting],
        "weighting_factor": [weighting_factor],
        "coverage": [coverage],
        "mse": [mse],
        "rmse": [rmse],
        "mae": [mae],
    }

    cur_res_df = pd.DataFrame(cur_res_dict)

    try:
        df = pd.read_csv("final_result_full.csv")
    except:
        df = pd.DataFrame({})
        df.to_csv("final_result_full.csv", index=False)

    df_final = pd.concat([df, cur_res_df])
    df_final.to_csv("final_result_full.csv", index=False)

    return (mse, rmse, mae), list(square(error_list))


def from_file_to_2D(path, genrefile, itemfile):
    """Load feature matrix from specified file
    Parameters:
    -- path: directory path to datafile and itemfile
    -- genrefile: delimited file that maps genre to genre index
    -- itemfile: delimited file that maps itemid to item name and genre

    Returns:
    -- movies: a dictionary containing movie titles (value) for a given movieID (key)
    -- genres: dictionary, key is genre, value is index into row of features array
    -- features: a 2D list of features by item, values are 1 and 0;
                 rows map to items and columns map to genre
                 returns as np.array()

    """
    # Get movie titles, place into movies dictionary indexed by itemID
    movies = {}
    try:
        with open(path + "/" + itemfile, encoding="iso8859") as myfile:
            # this encoding is required for some datasets: encoding='iso8859'
            for line in myfile:
                (id, title) = line.split("|")[0:2]
                movies[id] = title.strip()

    # Error processing
    except UnicodeDecodeError as ex:
        print(ex)
        print(len(movies), line, id, title)
        return {}
    except ValueError as ex:
        print("ValueError", ex)
        print(len(movies), line, id, title)
    except Exception as ex:
        print(ex)
        print(len(movies))
        return {}

    ##
    # Get movie genre from the genre file, place into genre dictionary indexed by genre index
    genres = {}  # key is genre index, value is the genre string
    ##
    try:
        with open(path + "/" + genrefile, encoding="iso8859") as myfile:
            # this encoding is required for some datasets: encoding='iso8859'
            for line in myfile:
                (genre, genre_index) = line.split("|")[0:2]
                genres[int(genre_index.strip())] = genre.replace("-", "")
    # Error processing
    except UnicodeDecodeError as ex:
        print(ex)
        print(len(genres), line, genre_index, genre)
        return {}
    except ValueError as ex:
        print("ValueError", ex)
        print(len(genres), line, genre_index, genre)
    except Exception as ex:
        print(ex)
        print(len(genres))
        return {}

    # Load data into a nested 2D list
    features = []
    start_feature_index = 5
    try:
        for line in open(path + "/" + itemfile, encoding="iso8859"):
            # print(line, line.split('|')) #debug
            fields = line.split("|")[start_feature_index:]
            row = []
            for feature in fields:
                row.append(int(feature))
            features.append(row)
        features = np.array(features)
    except Exception as ex:
        print(ex)
        print("Proceeding with len(features)", len(features))
        # return {}

    # return features matrix
    return movies, genres, features


def from_file_to_dict(path, datafile, itemfile):
    """Load user-item matrix from specified file

    Parameters:
    -- path: directory path to datafile and itemfile
    -- datafile: delimited file containing userid, itemid, rating
    -- itemfile: delimited file that maps itemid to item name

    Returns:
    -- prefs: a nested dictionary containing item ratings (value) for each user (key)

    """

    # Get movie titles, place into movies dictionary indexed by itemID
    movies = {}
    try:
        with open(path + "/" + itemfile, encoding="iso8859") as myfile:
            # this encoding is required for some datasets: encoding='iso8859'
            for line in myfile:
                (id, title) = line.split("|")[0:2]
                movies[id] = title.strip()

    # Error processing
    except UnicodeDecodeError as ex:
        print(ex)
        print(len(movies), line, id, title)
        return {}
    except ValueError as ex:
        print("ValueError", ex)
        print(len(movies), line, id, title)
    except Exception as ex:
        print(ex)
        print(len(movies))
        return {}

    # Load data into a nested dictionary
    prefs = {}
    for line in open(path + "/" + datafile):
        # print(line, line.split('\t')) #debug
        (user, movieid, rating, ts) = line.split("\t")
        user = user.strip()  # remove spaces
        movieid = movieid.strip()  # remove spaces
        prefs.setdefault(user, {})  # make it a nested dicitonary
        prefs[user][movies[movieid]] = float(rating)

    # return a dictionary of preferences
    return prefs


def transformPrefs(prefs):
    result = {}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item, {})

            # Flip item and person
            result[item][person] = prefs[person][item]
    return result


def prefs_to_2D_list(prefs):
    """
    Convert prefs dictionary into 2D list used as input for the MF class

    Parameters:
        prefs: user-item matrix as a dicitonary (dictionary)

    Returns:
        ui_matrix: (list) contains user-item matrix as a 2D list

    """
    ui_matrix = []

    user_keys_list = list(prefs.keys())
    num_users = len(user_keys_list)
    # print (len(user_keys_list), user_keys_list[:10]) # debug

    itemPrefs = transformPrefs(prefs)  # traspose the prefs u-i matrix
    item_keys_list = list(itemPrefs.keys())
    num_items = len(item_keys_list)
    # print (len(item_keys_list), item_keys_list[:10]) # debug

    sorted_list = True  # <== set manually to test how this affects results

    if sorted_list == True:
        user_keys_list.sort()
        item_keys_list.sort()
        print("\nsorted_list =", sorted_list)

    # initialize a 2D matrix as a list of zeroes with
    #     num users (height) and num items (width)

    for i in range(num_users):
        row = []
        for j in range(num_items):
            row.append(0.0)
        ui_matrix.append(row)

    # populate 2D list from prefs
    # Load data into a nested list

    for user in prefs:
        for item in prefs[user]:
            user_idx = user_keys_list.index(user)
            movieid_idx = item_keys_list.index(item)

            try:
                # make it a nested list
                ui_matrix[user_idx][movieid_idx] = prefs[user][item]
            except Exception as ex:
                print(ex)
                print(user_idx, movieid_idx)

    # return 2D user-item matrix
    return ui_matrix


def to_array(prefs):
    """convert prefs dictionary into 2D list"""
    R = prefs_to_2D_list(prefs)
    R = np.array(R)
    print("to_array -- height: %d, width: %d" % (len(R), len(R[0])))
    return R


def to_string(features):
    """convert features np.array into list of feature strings"""

    feature_str = []
    for i in range(len(features)):
        row = ""
        for j in range(len(features[0])):
            row += str(features[i][j])
        feature_str.append(row)
    print(
        "to_string -- height: %d, width: %d" % (len(feature_str), len(feature_str[0]))
    )
    return feature_str


def to_docs(features_str, genres):
    """convert feature strings to a list of doc strings for TFIDF"""

    feature_docs = []
    for doc_str in features_str:
        row = ""
        for i in range(len(doc_str)):
            if doc_str[i] == "1":
                row += genres[i] + " "  # map the indices to the actual genre string
        feature_docs.append(row.strip())  # and remove that pesky space at the end

    print("to_docs -- height: %d, width: varies" % (len(feature_docs)))
    return feature_docs


def cosine_sim(docs):
    """Simple example of cosine sim calcs"""
    print()
    print("## Cosine Similarity calc ##")
    print()
    # print("Documents:")  # , docs)
    # for i in range(len(docs)):
    #     print("%d: %s" % (i + 1, docs[i]))

    """ These are the key steps to calc the cosine sim matrix """

    # STEP 1: TF-IDF calc
    tfidf_vectorizer = TfidfVectorizer(
        lowercase=True, smooth_idf=True, norm="l2"
    )  # instantiate default -- SP22
    tfidf_matrix = tfidf_vectorizer.fit_transform(
        docs
    )  # get sklearn tfidf weights into a matrix
    # STEP 2: Cosine Sim matrix calc
    cosim = cosine_similarity(tfidf_matrix[0:], tfidf_matrix)

    return cosim


def movie_to_ID(movies):
    '''converts movies mapping from "id to title" to "title to id"'''
    return {x[1]: x[0] for x in movies.items()}


def get_TFIDF_recommendations(prefs, cosim_matrix, user, n, movies, threshold):
    """
    Calculates recommendations for a given user

    Parameters:
    -- prefs: dictionary containing user-item matrix
    -- cosim_matrix: list containing item_feature-item_feature cosine similarity matrix
    -- user: string containing name of user requesting recommendation

    Returns:
    -- ranknigs: A list of recommended items with 0 or more tuples,
       each tuple contains (predicted rating, item name).
       List is sorted, high to low, by predicted rating.
       An empty list is returned when no recommendations have been calc'd.

    """
    # find more details in Final Project Specification
    movies_inv = movie_to_ID(movies)
    scores = {}
    totalSim = {}

    # loop over items the user has rated
    for (item, rating) in prefs[user].items():
        item_index = int(movies_inv[item]) - 1

        # Loop over items similar to this one
        for j, sim in enumerate(cosim_matrix[item_index]):
            similar_item = movies[str(j + 1)]

            # make sure we pass threshold test and that
            # the item has not already been rated by the user
            if (sim >= threshold) and not (similar_item in prefs[user]):

                # Weighted sum of rating times similarity
                scores.setdefault(similar_item, 0)
                scores[similar_item] += sim * rating
                # Sum of all the similarities
                totalSim.setdefault(similar_item, 0)
                totalSim[similar_item] += sim

    # Divide each total score by total weighting to get an average
    rankings = [
        (score / totalSim[item], item)
        for item, score in scores.items()
        if totalSim[item] > 0
    ]

    # Return the rankings from highest to lowest
    rankings.sort()
    rankings.reverse()
    return rankings[:n]


def new_get_TFIDF_recommendations(
    prefs, cosim_matrix, user, threshold, cur_item, movies
):
    """
    Calculates recommendations for a given user

    Parameters:
    -- prefs: dictionary containing user-item matrix
    -- cosim_matrix: list containing item_feature-item_feature cosine similarity matrix
    -- user: string containing name of user requesting recommendation

    Returns:
    -- ranknigs: A list of recommended items with 0 or more tuples,
       each tuple contains (predicted rating, item name).
       List is sorted, high to low, by predicted rating.
       An empty list is returned when no recommendations have been calc'd.

    """
    # find more details in Final Project Specification
    movies_inv = movie_to_ID(movies)
    scores = {}
    totalSim = {}
    userRatings = prefs[user]

    cur_item_index = int(movies_inv[cur_item]) - 1
    
    # print(f'User ratings: {userRatings}')

    # print(f'Making prediction for {user} for {cur_item}')
    # loop through items this user has rated
    for item in userRatings:
        rating = userRatings[item]


        item_index = int(movies_inv[item]) - 1
        sim = cosim_matrix[item_index][cur_item_index]
        # print(f'Curr rating: {rating}, Curr sim: {sim}')

        # make sure we pass threshold test
        if sim <= threshold:
            continue
    
        # Weighted sum of rating times similarity
        scores.setdefault(cur_item, 0)
        scores[cur_item] += sim * rating
        # Sum of all the similarities
        totalSim.setdefault(cur_item, 0)
        totalSim[cur_item] += sim

    # Divide each total score by total weighting to get an average
    # for item, score in scores.items():
    rankings = [
        (score / totalSim[item], item)
        for item, score in scores.items()
        if cur_item == item
    ]
    # print(f'Rankings: {rankings}')
    # Return the rankings from highest to lowest
    rankings.sort()
    rankings.reverse()
    return rankings


def itemsim_to_np_matrix(itemsim, movies):
    n_items = len(movies)
    inv_movies = movie_to_ID(movies)
    np_arr = np.zeros((n_items, n_items))

    for movie in inv_movies:
        i = int(inv_movies[movie]) - 1

        if movie not in itemsim:
            continue

        for sim, sim_movie in itemsim[movie]:
            j = int(inv_movies[sim_movie]) - 1
            np_arr[i][j] = sim

    return np_arr


def hybrid_update_sim_matrix(cosim_matrix, item_item_matrix, weighting_factor):
    """
    Updates the TFIDF sim matrix for the hybrid recommender system based on the weighting factor

    Parameters:
    -- cosim_matrix: list containing item_feature-item_feature cosine similarity matrix
    -- item_item_matrix: list containing item-item similarity matrix based on Pearson or Euclidean
    -- weighting_factor: factor to be used to update the TFIDF sim matrix (0, 0.25, 0.5, 0.75, or 1)

    Returns:
    -- updated_cosim_matrix: list containing item_feature-item_feature cosine similarity matrix where
                             0s have been replaced according to item-item matrix
    """
    print(f"cosim: {np.array(cosim_matrix).shape}")
    print(f"item: {np.array(item_item_matrix).shape}")
    copy = cosim_matrix.copy()

    for i in range(len(cosim_matrix)):
        for j in range(len(cosim_matrix[i])):
            if cosim_matrix[i][j] == 0:
                copy[i][j] = item_item_matrix[i][j] * weighting_factor

    return copy


def get_hybrid_recommendations(prefs, updated_cosim_matrix, user, n, movies, threshold):
    """
    Calculates recommendations for a given user

    Parameters:
    -- prefs: dictionary containing user-item matrix
    -- updated_cosim_matrix: list containing item_feature-item_feature cosine similarity matrix where
                             0s have been replaced according to item-item matrix
    -- user: string containing name of user requesting recommendation

    Returns:
    -- rankings: A list of recommended items with 0 or more tuples,
       each tuple contains (predicted rating, item name).
       List is sorted, high to low, by predicted rating.
       An empty list is returned when no recommendations have been calc'd.

    """
    # find more details in Final Project Specification
    movies_inv = movie_to_ID(movies)
    scores = {}
    totalSim = {}

    # loop over items the user has rated
    for (item, rating) in prefs[user].items():
        item_index = int(movies_inv[item]) - 1

        # Loop over items similar to this one
        for j, sim in enumerate(updated_cosim_matrix[item_index]):
            similar_item = movies[str(j + 1)]

            # make sure we pass threshold test and that
            # the item has not already been rated by the user
            if (sim >= threshold) and not (similar_item in prefs[user]):

                # Weighted sum of rating times similarity
                scores.setdefault(similar_item, 0)
                scores[similar_item] += sim * rating
                # Sum of all the similarities
                totalSim.setdefault(similar_item, 0)
                totalSim[similar_item] += sim

    # Divide each total score by total weighting to get an average
    rankings = [
        (score / totalSim[item], item)
        for item, score in scores.items()
        if totalSim[item] > 0
    ]

    # Return the rankings from highest to lowest
    rankings.sort()
    rankings.reverse()
    return rankings[:n]


def new_get_hybrid_recommendations(
    prefs, updated_cosim_matrix, user, threshold, cur_item, movies
):
    """
    Calculates recommendations for a given user

    Parameters:
    -- prefs: dictionary containing user-item matrix
    -- updated_cosim_matrix: list containing item_feature-item_feature cosine similarity matrix where
                             0s have been replaced according to item-item matrix
    -- user: string containing name of user requesting recommendation

    Returns:
    -- rankings: A list of recommended items with 0 or more tuples,
       each tuple contains (predicted rating, item name).
       List is sorted, high to low, by predicted rating.
       An empty list is returned when no recommendations have been calc'd.

    """
    # find more details in Final Project Specification
    movies_inv = movie_to_ID(movies)
    scores = {}
    totalSim = {}
    userRatings = prefs[user]

    cur_item_index = int(movies_inv[cur_item]) - 1

    # loop through items this user has rated
    for item in userRatings:
        rating = userRatings[item]
        item_index = int(movies_inv[item]) - 1
        sim = updated_cosim_matrix[item_index][cur_item_index]

        # make sure we pass threshold test and that
        # the item has not already been rated by the user
        if sim >= threshold:
            # Weighted sum of rating times similarity
            scores.setdefault(cur_item, 0)
            scores[cur_item] += sim * rating
            # Sum of all the similarities
            totalSim.setdefault(cur_item, 0)
            totalSim[cur_item] += sim

    # Divide each total score by total weighting to get an average
    rankings = [
        (score / totalSim[item], item)
        for item, score in scores.items()
        if totalSim[item] > 0
    ]

    # Return the rankings from highest to lowest
    rankings.sort()
    rankings.reverse()
    return rankings


class ExplicitMF():
    def __init__(self, 
                 ratings,
                 n_factors=40,
                 learning='sgd',
                 sgd_alpha = 0.1,
                 sgd_beta = 0.1,
                 sgd_random = False,
                 item_fact_reg=0.0, 
                 user_fact_reg=0.0,
                 item_bias_reg=0.0,
                 user_bias_reg=0.0,
                 max_iters = 20,
                 verbose=True):
        """
        Train a matrix factorization model to predict empty 
        entries in a matrix. The terminology assumes a 
        ratings matrix which is ~ user x item
        
        Params
        ======
        ratings : (ndarray)
            User x Item matrix with corresponding ratings
            Note: can be full ratings matrix or train matrix
        
        n_factors : (int)
            Number of latent factors to use in matrix factorization model
            
        learning : (str)
            Method of optimization. Options include 'sgd' or 'als'.
        
        sgd_alpha: (float)
            learning rate for sgd
            
        sgd_beta:  (float)
            regularization for sgd
            
        sgd_random: (boolean)
            False makes use of random.seed(0)
            False means don't make it random (ie, make it predictable)
            True means make it random (ie, changee everytime code is run)
        
        item_fact_reg : (float)
            Regularization term for item latent factors
            Note: currently, same value as user_fact_reg
        
        user_fact_reg : (float)
            Regularization term for user latent factors
            Note: currently, same value as item_fact_reg
            
        item_bias_reg : (float)
            Regularization term for item biases
            Note: for later use, not used currently
        
        user_bias_reg : (float)
            Regularization term for user biases
            Note: for later use, not used currently
            
        max_iters : (integer)
            maximum number of iterations
        
        verbose : (bool)
            Whether or not to printout training progress
            
            
        Original Source info: 
            https://blog.insightdatascience.com/explicit-matrix-factorization-als-sgd-and-all-that-jazz-b00e4d9b21ea#introsgd
            https://gist.github.com/EthanRosenthal/a293bfe8bbe40d5d0995#file-explicitmf-py
        """
        
        self.ratings = ratings 
        self.n_users, self.n_items = ratings.shape
        self.n_factors = n_factors
        self.item_fact_reg = item_fact_reg
        self.user_fact_reg = user_fact_reg
        self.item_bias_reg = item_bias_reg 
        self.user_bias_reg = user_bias_reg 
        self.learning = learning
        if self.learning == 'als':
            np.random.seed(0)
        if self.learning == 'sgd':
            self.sample_row, self.sample_col = self.ratings.nonzero()
            self.n_samples = len(self.sample_row)
            self.sgd_alpha = sgd_alpha # sgd learning rate, alpha
            self.sgd_beta = sgd_beta # sgd regularization, beta
            self.sgd_random = sgd_random # randomize
            if self.sgd_random == False:
                np.random.seed(0) # do not randomize
        self._v = verbose
        self.max_iters = max_iters
        self.nonZero = ratings > 0 # actual values
        
        print()
        if self.learning == 'als':
            print('ALS instance parameters:\nn_factors=%d, user_reg=%.5f,  item_reg=%.5f, num_iters=%d' %\
              (self.n_factors, self.user_fact_reg, self.item_fact_reg, self.max_iters))
        
        elif self.learning == 'sgd':
            print('SGD instance parameters:\nnum_factors K=%d, learn_rate alpha=%.5f, reg beta=%.5f, num_iters=%d, sgd_random=%s' %\
              (self.n_factors, self.sgd_alpha, self.sgd_beta, self.max_iters, self.sgd_random ) )
        print()

    def train(self, n_iter=10): 
        """ Train model for n_iter iterations from scratch."""
        
        def normalize_row(x):
            norm_row =  x / sum(x) # weighted values: each row adds up to 1
            return norm_row

        # initialize latent vectors        
        self.user_vecs = np.random.normal(scale=1./self.n_factors,\
                                          size=(self.n_users, self.n_factors))
        self.item_vecs = np.random.normal(scale=1./self.n_factors,
                                          size=(self.n_items, self.n_factors))
        
        if self.learning == 'als':
            ## Try one of these. apply_long_axis came from Explicit_RS_MF_ALS()
            ##                                             Daniel Nee code
            
            self.user_vecs = abs(np.random.randn(self.n_users, self.n_factors))
            self.item_vecs = abs(np.random.randn(self.n_items, self.n_factors))
            
            #self.user_vecs = np.apply_along_axis(normalize_row, 1, self.user_vecs) # axis=1, across rows
            #self.item_vecs = np.apply_along_axis(normalize_row, 1, self.item_vecs) # axis=1, across rows
            
            self.partial_train(n_iter)
            
        elif self.learning == 'sgd':
            self.user_bias = np.zeros(self.n_users)
            self.item_bias = np.zeros(self.n_items)
            self.global_bias = np.mean(self.ratings[np.where(self.ratings != 0)])
            self.partial_train(n_iter)
    
    def partial_train(self, n_iter):
        """ 
        Train model for n_iter iterations. 
        Can be called multiple times for further training.
        Remains in the while loop for a number of iterations, calculated from
        the contents of the iter_array in calculate_learning_curve()
        """
        
        ctr = 1
        while ctr <= n_iter:

            if self.learning == 'als':
                self.user_vecs = self.als_step(self.user_vecs, 
                                               self.item_vecs, 
                                               self.ratings, 
                                               self.user_fact_reg, 
                                               type='user')
                self.item_vecs = self.als_step(self.item_vecs, 
                                               self.user_vecs, 
                                               self.ratings, 
                                               self.item_fact_reg, 
                                               type='item')
                
            elif self.learning == 'sgd':
                self.training_indices = np.arange(self.n_samples)
                np.random.shuffle(self.training_indices)
                self.sgd()
            ctr += 1

    def als_step(self,
                 latent_vectors,
                 fixed_vecs,
                 ratings,
                 _lambda,
                 type='user'):
        """
        ALS algo step.
        Solve for the latent vectors specified by type parameter: user or item
        """
        
        #lv_shape = latent_vectors.shape[0] ## debug
        
        if type == 'user':

            for u in range(latent_vectors.shape[0]): # latent_vecs ==> user_vecs
                #r_u = ratings[u, :] ## debug
                #fvT = fixed_vecs.T ## debug
                idx = self.nonZero[u,:] # get the uth user profile with booleans 
                                        # (True when there are ratings) based on 
                                        # ratingsMatrix, n x 1
                nz_fixed_vecs = fixed_vecs[idx,] # get the item vector entries, non-zero's x f
                YTY = nz_fixed_vecs.T.dot(nz_fixed_vecs) # fixed_vecs are item_vecs
                lambdaI = np.eye(YTY.shape[0]) * _lambda
                
                latent_vectors[u, :] = \
                    solve( (YTY + lambdaI) , nz_fixed_vecs.T.dot (ratings[u, idx] ) )

                '''
                ## debug
                if u <= 10: 
                    print('user vecs1', nz_fixed_vecs)
                    print('user vecs1', fixed_vecs, '\n', ratings[u, :] )
                    print('user vecs2', fixed_vecs.T.dot (ratings[u, :] ))
                    print('reg', YTY, '\n', lambdaI)
                    print('new user vecs:\n', latent_vectors[u, :])
                ## debug
                '''
                    
        elif type == 'item':
            
            for i in range(latent_vectors.shape[0]): #latent_vecs ==> item_vecs
                idx = self.nonZero[:,i] # get the ith item "profile" with booleans 
                                        # (True when there are ratings) based on 
                                        # ratingsMatrix, n x 1
                nz_fixed_vecs = fixed_vecs[idx,] # get the item vector entries, non-zero's x f
                XTX = nz_fixed_vecs.T.dot(nz_fixed_vecs) # fixed_vecs are user_vecs
                lambdaI = np.eye(XTX.shape[0]) * _lambda
                latent_vectors[i, :] = \
                    solve( (XTX + lambdaI) , nz_fixed_vecs.T.dot (ratings[idx, i] ) )

        return latent_vectors

    def sgd(self):
        ''' run sgd algo '''
        
        for idx in self.training_indices:
            u = self.sample_row[idx]
            i = self.sample_col[idx]
            prediction = self.predict(u, i)
            e = (self.ratings[u,i] - prediction) # error
            
            # Update biases
            self.user_bias[u] += self.sgd_alpha * \
                                (e - self.sgd_beta * self.user_bias[u])
            self.item_bias[i] += self.sgd_alpha * \
                                (e - self.sgd_beta * self.item_bias[i])
            
            # Create copy of row of user_vecs since we need to update it but
            #    use older values for update on item_vecs, 
            #    so make a deepcopy of previous user_vecs
            previous_user_vecs = deepcopy(self.user_vecs[u, :])
            
            # Update latent factors
            self.user_vecs[u, :] += self.sgd_alpha * \
                                    (e * self.item_vecs[i, :] - \
                                     self.sgd_beta * self.user_vecs[u,:])
            self.item_vecs[i, :] += self.sgd_alpha * \
                                    (e * previous_user_vecs - \
                                     self.sgd_beta * self.item_vecs[i,:])           
    
    def calculate_learning_curve(self, iter_array, test):
        """
        Keep track of MSE as a function of training iterations.
        
        Params
        ======
        iter_array : (list)
            List of numbers of iterations to train for each step of 
            the learning curve. e.g. [1, 5, 10, 20]
        test : (2D ndarray)
            Testing dataset (assumed to be user x item)
        
        
        
        This function creates two new class attributes:
        
        train_mse : (list)
            Training data MSE values for each value of iter_array
        test_mse : (list)
            Test data MSE values for each value of iter_array
        """
        
        print()
        if self.learning == 'als':
            print('Runtime parameters:\nn_factors=%d, user_reg=%.5f, item_reg=%.5f,'
                  ' max_iters=%d,'
                  ' \nratings matrix: %d users X %d items' %\
                  (self.n_factors, self.user_fact_reg, self.item_fact_reg, 
                   self.max_iters, self.n_users, self.n_items))
        if self.learning == 'sgd':
            print('Runtime parameters:\nn_factors=%d, learning_rate alpha=%.3f,'
                  ' reg beta=%.5f, max_iters=%d, sgd_random=%s'
                  ' \nratings matrix: %d users X %d items' %\
                  (self.n_factors, self.sgd_alpha, self.sgd_beta, 
                   self.max_iters, self.sgd_random, self.n_users, self.n_items))
        print()       
        
        iter_array.sort()
        self.train_mse =[]
        self.test_mse = []
        iter_diff = 0
        
        start_time = time()
        stop_time = time()
        elapsed_time = (stop_time-start_time) #/60
        print ( 'Elapsed train/test time %.2f secs' % elapsed_time )        
        
        # Loop through number of iterations
        for (i, n_iter) in enumerate(iter_array):
            if self._v:
                print ('Iteration: {}'.format(n_iter))
            if i == 0:
                self.train(n_iter - iter_diff) # init training, run first iter
            else:
                self.partial_train(n_iter - iter_diff) # run more iterations
                    # .. as you go from one element of iter_array to another

            predictions = self.predict_all() # calc dot product of p and qT
            # calc train  errors -- predicted vs actual
            self.train_mse += [self.get_mse(predictions, self.ratings)]
            if test.any() > 0: # check if test matrix is all zeroes ==> Train Only
                               # If so, do not calc mse and avoid runtime error   
                # calc test errors -- predicted vs actual 
                self.test_mse += [self.get_mse(predictions, test)]
            else:
                self.test_mse = ['n/a']
            if self._v:
                print ('Train mse: ' + str(self.train_mse[-1]))
                if self.test_mse != ['n/a']:
                    print ('Test mse: ' + str(self.test_mse[-1]))
            iter_diff = n_iter
            
            stop_time = time()
            elapsed_time = (stop_time-start_time) #/60
            print ( 'Elapsed train/test time %.2f secs' % elapsed_time )     

    def predict(self, u, i):
        """ Single user and item prediction """
        
        if self.learning == 'als':
            return self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
        elif self.learning == 'sgd':
            prediction = self.global_bias + self.user_bias[u] + self.item_bias[i]
            prediction += self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
            return prediction
    
    def predict_all(self):
        """ Predict ratings for every user and item """
        
        predictions = np.zeros((self.user_vecs.shape[0], 
                                self.item_vecs.shape[0]))
        for u in range(self.user_vecs.shape[0]):
            for i in range(self.item_vecs.shape[0]):
                predictions[u, i] = self.predict(u, i)
        return predictions    

    def get_mse(self, pred, actual):
        ''' Calc MSE between predicted and actual values '''
        
        # Ignore nonzero terms.
        pred = pred[actual.nonzero()].flatten()
        actual = actual[actual.nonzero()].flatten()
        return mean_squared_error(pred, actual)


def ratings_to_2D_matrix(ratings, m, n):
    '''
    creates a U-I matrix from the data
    ==>>  eliminates movies (items) that have no ratings!
    '''
    print('Summary Stats:')
    print()
    print(ratings.describe())
    ratingsMatrix = ratings.pivot_table(columns=['item_id'], index =['user_id'],
        values='rating', dropna = False) # convert to a U-I matrix format from file input
    ratingsMatrix = ratingsMatrix.fillna(0).values # replace nan's with zeroes
    ratingsMatrix = ratingsMatrix[0:m,0:n] # get rid of any users/items that have no ratings
    print()
    print('2D_matrix shape', ratingsMatrix.shape) # debug
    
    return ratingsMatrix


def file_info(df):
    ''' print file info/stats  '''
    print()
    print (df.head())
    
    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]
    
    ratings = ratings_to_2D_matrix(df, n_users, n_items)
    
    print()
    print (ratings)
    print()
    print (str(n_users) + ' users')
    print (str(n_items) + ' items')
    
    sparsity = float(len(ratings.nonzero()[0]))
    sparsity /= (ratings.shape[0] * ratings.shape[1])
    sparsity *= 100
    sparsity = 100 - sparsity
    print ('Sparsity: {:4.2f}%'.format(sparsity))
    return ratings


def train_test_split(ratings, TRAIN_ONLY):
    ''' split the data into train and test '''
    test = np.zeros(ratings.shape)
    train = deepcopy(ratings) # instead of copy()
    
    ## setting the size parameter for random.choice() based on dataset size
    if len(ratings) < 10: # critics
        size = 1
    elif len(ratings) < 1000: # ml-100k
        size = 20
    else:
        size = 40 # ml-1m
        
    #print('size =', size) ## debug
    
    if TRAIN_ONLY == False:
        np.random.seed(0) # do not randomize the random.choice() in this function,
                          # let ALS or SGD make the decision to randomize
                          # Note: this decision can be reset with np.random.seed()
                          # .. see code at the end of this for loop
        for user in range(ratings.shape[0]): ## CES changed all xrange to range for Python v3
            test_ratings = np.random.choice(ratings[user, :].nonzero()[0], 
                                            size=size, 
                                            replace=True) #False)
            # When replace=False, size for ml-100k = 20, for critics = 1,2, or 3
            # Use replace=True for "better" results
            
            '''
            np.random.choice() info ..
            https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
            
            random.choice(a, size=None, replace=True, p=None)
            
            Parameters --
            a:         1-D array-like or int
            If an ndarray, a random sample is generated from its elements. 
            If an int, the random sample is generated as if it were np.arange(a)
            
            size:      int or tuple of ints, optional
            Output shape. If the given shape is, e.g., (m, n, k), 
            then m * n * k samples are drawn. 
            Default is None, in which case a single value is returned.
        
            replace:   boolean, optional
            Whether the sample is with or without replacement. 
            Default is True, meaning that a value of a can be selected multiple times.
        
            p:        1-D array-like, optional
            The probabilities associated with each entry in a. If not given, 
            the sample assumes a uniform distribution over all entries in a.
    
            Returns
            samples:   single item or ndarray
            The generated random samples
            
            '''
            
            train[user, test_ratings] = 0.
            test[user, test_ratings] = ratings[user, test_ratings]
            
        # Test and training are truly disjoint
        assert(np.all((train * test) == 0)) 
        np.random.seed() # allow other functions to randomize
    
    #print('TRAIN_ONLY (in split) =', TRAIN_ONLY) ##debug
    
    return train, test


def train_test_validation_split(dataset):
    train, test_temp = sklearn.model_selection.train_test_split(dataset, test_size=0.2, random_state=42)
    val, test = sklearn.model_selection.train_test_split(test_temp, test_size=0.5, random_state=42)

    return train, test, val


def test_train_info(test, train):
    ''' print test/train info   '''

    print()
    print ('Train info: %d rows, %d cols' % (len(train), len(train[0])))
    print ('Test info: %d rows, %d cols' % (len(test), len(test[0])))
    
    test_count = 0
    for i in range(len(test)):
        for j in range(len(test[0])):
            if test[i][j] !=0:
                test_count += 1
                #print (i,j,test[i][j]) # debug
    print('test ratings count =', test_count)
    
    train_count = 0
    for i in range(len(train)):
        for j in range(len(train[0])):
            if train[i][j] !=0:
                train_count += 1
                #print (i,j,train[i][j]) # debug
    
    total_count = test_count + train_count
    print('train ratings count =', train_count)
    print('test + train count', total_count)
    print('test/train percentages: %0.2f / %0.2f' 
          % ( (test_count/total_count)*100, (train_count/total_count)*100 ))
    print()


def plot_learning_curve(iter_array, model, parms, dataset):
    ''' plot the error curve '''
    
    ## Note: the iter_array can cause plots to NOT 
    ##    be smooth! If matplotlib can't smooth, 
    ##    then print/plot results every 
    ##    max_num_iterations/10 (rounded up)
    ##    instead of using an iter_array list
    
    #print('model.test_mse', model.test_mse) # debug
    if model.test_mse != ['n/a']:
        plt.plot(iter_array, model.test_mse, label='Test', linewidth=3)
    plt.plot(iter_array, model.train_mse, label='Train', linewidth=3)

    plt.xticks(fontsize=10); # 16
    plt.xticks(iter_array, iter_array)
    plt.yticks(fontsize=10);
    
    axes = plt.gca()
    axes.grid(True) # turns on grid
    
    if model.learning == 'als':
        runtime_parms = \
            'shape=%s, n_factors=%d, user_fact_reg=%.3f, item_fact_reg=%.3f'%\
            (model.ratings.shape, model.n_factors, model.user_fact_reg, model.item_fact_reg)
            #(train.shape, model.n_factors, model.user_fact_reg, model.item_fact_reg)
        plt.title("ALS Model Evaluation\n%s" % runtime_parms , fontsize=10) 
    elif model.learning == 'sgd':
        runtime_parms = \
            'shape=%s, num_factors K=%d, alpha=%.3f, beta=%.3f'%\
            (model.ratings.shape, model.n_factors, model.sgd_alpha, model.sgd_beta)
            #(train.shape, model.n_factors, model.learning_rate, model.user_fact_reg)
        plt.title("SGD Model Evaluation\n%s" % runtime_parms , fontsize=10)         
    
    plt.xlabel('Iterations', fontsize=15);
    plt.ylabel('Mean Squared Error', fontsize=15);
    plt.legend(loc='best', fontsize=15, shadow=True) # 'best', 'center right' 20
    
    if model.learning == 'als':
        plt.savefig(f'results_als_{dataset}/ALS_{dataset}_{parms[0]}_{parms[1]}_{parms[2]}_plot.png')
    elif model.learning == 'sgd':
        plt.savefig(f'results_sgd_{dataset}/SGD_{dataset}_{parms[0]}_{parms[1]}_{parms[2]}_{parms[3]}_plot.png')
    plt.close()


def main():
    """User interface for Python console"""

    # Load critics dict from file
    path = os.getcwd()  # this gets the current working directory
    # you can customize path for your own computer here
    print("\npath: %s" % path)  # debug
    done = False
    prefs = {}
    pd_rml_ran = False
    pd_r_ran = False
    ttv_ran = False
    ncf_built = False
    ncf_trained = False

    while not done:
        print()
        # Start a simple dialog
        file_io = input(
            "R(ead) critics data from file?, \n"
            'PD-R(ead) critics data from file?, \n'
            "RML(ead) Movie Lens data from file?, \n"
            'PD-RML(Read) ml100K data from file?, \n'
            "P(rint) the U-I matrix?, \n"
            "V(alidate) the dictionary?, \n"
            "S(tats) print?, \n"
            "D(istance) critics data?, \n"
            "PC(earson Correlation) critics data?, \n"
            "U(ser-based CF Recommendations)?, \n"
            "LCV(eave one out cross-validation)? \n"
            "Sim(ilarity matrix) calc for Item-based recommender? \n"
            "Simu(ser similarity matrix) calc for User-based recommender? \n"
            'T(est/train datasets?, \n'
            'MF-ALS(Matrix factorization - Alternating Least Squares)? \n'
            'MF-SGD(Matrix factorization - Stochastic Gradient Descent)? \n'
            'MF-ALS-GRID(Matrix factorization - ALS - Grid Search)? \n'
            'MF-SGD-GRID(Matrix factorization - SGD - Grid Search)? \n'
            "I(tem-based CF Recommendations)?, \n"
            "LCVSIM(eave one out cross-validation)?, \n"
            "REC(ommendations, Item or User CF)?, \n"
            "TFIDF(and cosine sim Setup)?, \n"
            "TFIDF-GRID \n"
            "HYB(RID setup?), \n"
            "HYB-GRID \n"
            "RECS(ecommendations -- all algos)?, \n"
            "==>> "
        )

        if file_io == "R" or file_io == "r":
            print()
            file_dir = "data/"
            datafile = (
                "critics_ratings.data"  # for userids use 'critics_ratings_userIDs.data'
            )
            itemfile = "critics_movies.item"
            genrefile = "critics_movies.genre"  # movie genre file
            dataset_name = "critics"
            print('Reading "%s" dictionary from file' % datafile)
            prefs = from_file_to_dict(path, file_dir + datafile, file_dir + itemfile)
            print('Prefs')
            print(prefs)
            movies, genres, features = from_file_to_2D(
                path, file_dir + genrefile, file_dir + itemfile
            )
            print(
                "Number of users: %d\nList of users:" % len(prefs), list(prefs.keys())
            )

            ##
            print(
                "Number of distinct genres: %d, number of feature profiles: %d"
                % (len(genres), len(features))
            )
            print("genres")
            print(genres)
            print("features")
            print(features)

            # reset these when dataset is loaded/reloaded
            tfidf_ran = False
            sim_ran = False
            hybrid_ran = False

        elif file_io == 'PD-R' or file_io == 'pd-r':
            
            # Load user-item matrix from file
            dataset = 'critics'
            ## Read in data: critics
            data_folder = '/data/' # for critics
            #print('\npath: %s\n' % path_name + data_folder) # debug: print path info
            names = ['user_id', 'item_id', 'rating', 'timestamp'] # column headings
            
            #Create pandas dataframe
            df = pd.read_csv(path + data_folder + 'critics_ratings_userIDs.data', sep='\t', names=names) # for critics
            ratings = file_info(df)
            
            # set test/train in case they were set by a previous file I/O command
            test_train_done = False
            pd_r_ran = True
            print()
            print('Test and Train arrays are empty!')
            print()

        elif file_io == "RML" or file_io == "rml":
            print()
            file_dir = "data/ml-100k/"  # path from current directory
            datafile = "u.data"  # ratngs file
            itemfile = "u.item"  # movie titles file
            genrefile = "u.genre"  # movie genre file
            dataset_name = "ml-100k"
            print('Reading "%s" dictionary from file' % datafile)
            prefs = from_file_to_dict(path, file_dir + datafile, file_dir + itemfile)
            movies, genres, features = from_file_to_2D(
                path, file_dir + genrefile, file_dir + itemfile
            )
            print('Movies:')
            print(movies)

            print()
            print('Movie to ID: ')
            print(movie_to_ID(movies))
            ##
            print(
                "Number of users: %d\nList of users [0:10]:" % len(prefs),
                list(prefs.keys())[0:10],
            )
            print(
                "Number of distinct genres: %d, number of feature profiles: %d"
                % (len(genres), len(features))
            )
            print("genres")
            print(genres)
            print("features")
            print(features)

            # reset these when dataset is loaded/reloaded
            tfidf_ran = False
            sim_ran = False
            hybrid_ran = False

        elif file_io == 'PD-RML' or file_io == 'pd-rml':
                    
            dataset = 'ml-100k'
            # Load user-item matrix from file
            ## Read in data: ml-100k
            data_folder = '/data/ml-100k/' # for ml-100k                   
            #print('\npath: %s\n' % path_name + data_folder) # debug: print path info
            names = ['user_id', 'item_id', 'rating', 'timestamp'] # column headings
    
            #Create pandas dataframe
            df = pd.read_csv(path + data_folder + 'u.data', sep='\t', names=names) # for ml-100k
            ratings = file_info(df)
            
            n_items = len(pd.unique(df['item_id']))
            n_users = len(pd.unique(df['user_id']))
            test_train_done = False
            pd_rml_ran = True
            print()
            print('Test and Train arrays are empty!')
            print()
       
        elif file_io == "P" or file_io == "p":
            # print the u-i matrix
            print()
            if len(prefs) > 0:
                print('Printing "%s" dictionary from file' % datafile)
                print("User-item matrix contents: user, item, rating")
                count_u = MAX_PRINT
                for user in prefs:
                    count_i = MAX_PRINT
                    if count_u == 0:
                        break
                    for item in prefs[user]:
                        if count_i == 0:
                            break
                        print(user, item, prefs[user][item])
                        count_i -= 1
                    count_u -= 1
            else:
                print("Empty dictionary, R(ead) in some data!")

        elif file_io == "V" or file_io == "v":
            print()
            if len(prefs) > 0 and len(prefs) < 10:
                # Validate the dictionary contents ..
                print('Validating "%s" dictionary from file' % datafile)
                print(
                    "critics['Lisa']['Lady in the Water'] =",
                    prefs["Lisa"]["Lady in the Water"],
                )  # ==> 2.5
                print("critics['Toby']:", prefs["Toby"])
                # ==> {'Snakes on a Plane': 4.5, 'You, Me and Dupree': 1.0,
                #      'Superman Returns': 4.0}
            elif len(prefs) > 10:
                print("No validation steps set up for ml-100k dataset.")
            else:
                print("Empty dictionary, R(ead) in some data!")

        elif file_io == "S" or file_io == "s":
            print()
            filename = "critics_ratings.data"
            if len(prefs) > 0:
                data_stats(prefs, filename)
                popular_items(prefs, filename)

            else:  # Make sure there is data  to process ..
                print("Empty dictionary, R(ead) in some data!")

        elif file_io == "D" or file_io == "d":
            print()
            if len(prefs) > 0 and len(prefs) < 10:
                print("Examples:")
                print(
                    "Distance sim Lisa & Gene:", sim_distance(prefs, "Lisa", "Gene")
                )  # 0.29429805508554946
                num = 1
                den = 1 + sqrt(
                    (2.5 - 3.0) ** 2
                    + (3.5 - 3.5) ** 2
                    + (3.0 - 1.5) ** 2
                    + (3.5 - 5.0) ** 2
                    + (3.0 - 3.0) ** 2
                    + (2.5 - 3.5) ** 2
                )
                print("Distance sim Lisa & Gene (check):", num / den)
                print(
                    "Distance sim Lisa & Michael:",
                    sim_distance(prefs, "Lisa", "Michael"),
                )  # 0.4721359549995794
                print()

                print("User-User distance similarities:")

                uu_sim_matrix = []
                users = list(prefs.keys())
                most_similar_pair = ["user1", "user2", 0]
                least_similar_pair = ["user1", "user2", float("inf")]

                # Make a UU-similariy matrix and find the most similar pair
                # and the least similar pair
                for user1_index in range(len(users)):
                    row = []
                    for user2_index in range(len(users)):

                        sim_dist_value = round(
                            sim_distance(prefs, users[user1_index], users[user2_index]),
                            3,
                        )
                        row.append(sim_dist_value)
                        if user1_index != user2_index:
                            if sim_dist_value > most_similar_pair[2]:
                                most_similar_pair = [
                                    users[user1_index],
                                    users[user2_index],
                                    sim_dist_value,
                                ]
                            if sim_dist_value < least_similar_pair[2]:
                                least_similar_pair = [
                                    users[user1_index],
                                    users[user2_index],
                                    sim_dist_value,
                                ]

                    uu_sim_matrix.append(row)

                # Create a pretty string of the UU-similarity matrix
                full_str = "\t"
                for user in users:
                    full_str += user + "\t"
                full_str += "\n"

                for row in range(len(uu_sim_matrix)):
                    row_str = str(users[row]) + "\t"
                    for col in range(len(uu_sim_matrix[row])):
                        if row != col:
                            row_str += str(uu_sim_matrix[row][col]) + "\t"
                        else:
                            row_str += "X\t"

                    full_str += row_str + "\n"

                print(full_str)

                print(
                    "Most Similar User-User Pair: {u1} and {u2} with a sim distance of {sim_dist_value}".format(
                        u1=most_similar_pair[0],
                        u2=most_similar_pair[1],
                        sim_dist_value=most_similar_pair[2],
                    )
                )

                print(
                    "Least Similar User-User Pair: {u1} and {u2} with a sim distance of {sim_dist_value}".format(
                        u1=least_similar_pair[0],
                        u2=least_similar_pair[1],
                        sim_dist_value=least_similar_pair[2],
                    )
                )
            elif len(prefs) > 10:
                print("This step is only for the smaller critics data.")
            else:
                print("Empty dictionary, R(ead) in some data!")

        elif file_io == "PC" or file_io == "pc":
            print()
            if len(prefs) > 0 and len(prefs) < 10:
                print("Example:")
                print(
                    "Pearson sim Lisa & Gene:", sim_pearson(prefs, "Lisa", "Gene")
                )  # 0.39605901719066977
                print("Pearson sim Mick & Gene:", sim_pearson(prefs, "Mick", "Gene"))
                print()

                print("Pearson for all users:")

                uu_sim_pearson_matrix = []
                users = list(prefs.keys())

                # Make a UU-similariy matrix
                for user1_index in range(len(users)):
                    row = []
                    for user2_index in range(len(users)):

                        sim_pearson_value = round(
                            sim_pearson(prefs, users[user1_index], users[user2_index]),
                            3,
                        )
                        row.append(sim_pearson_value)

                    uu_sim_pearson_matrix.append(row)

                # Create a pretty string of the UU-similarity matrix
                full_str = "\t"
                for user in users:
                    full_str += user + "\t"
                full_str += "\n"

                for row in range(len(uu_sim_pearson_matrix)):
                    row_str = str(users[row]) + "\t"
                    for col in range(len(uu_sim_pearson_matrix[row])):
                        if row != col:
                            row_str += str(uu_sim_pearson_matrix[row][col]) + "\t"
                        else:
                            row_str += "X\t"

                    full_str += row_str + "\n"

                print(full_str)
            elif len(prefs) > 10:
                print("This step is only for the smaller critics data.")
            else:
                print("Empty dictionary, R(ead) in some data!")

        elif file_io == "U" or file_io == "u":
            print()
            if len(prefs) > 0 and len(prefs) < 10:
                print("Example:")
                user_name = "Toby"
                print(
                    "User-based CF recs for %s, sim_pearson: " % (user_name),
                    getRecommendations(prefs, user_name, similarity=sim_pearson),
                )

                print(
                    "User-based CF recs for %s, sim_distance: " % (user_name),
                    getRecommendations(prefs, user_name, similarity=sim_distance),
                )

                print()

            if len(prefs) > 0:
                print("User-based CF recommendations for all users:")

                get_all_UU_recs(prefs, sim=sim_pearson, num_users=10, top_N=5)

        elif file_io == "LCV" or file_io == "lcv":
            print()
            if len(prefs) > 0:
                print("LOO_CV Evaluation")

                error, error_list = loo_cv(
                    prefs, "MSE", sim_pearson, getRecommendations, dataset_name
                )
                print()
                error, error_list = loo_cv(
                    prefs, "MSE", sim_distance, getRecommendations, dataset_name
                )
            else:
                print("Empty dictionary, R(ead) in some data!")

        elif file_io == "Sim" or file_io == "sim":
            print()
            if len(prefs) > 0:
                ready = False  # sub command in progress
                sub_cmd = input(
                    "RD(ead) distance,\n"
                    "WD(rite) distance,\n"
                    "RP(ead) pearson,\n"
                    "WP(rite) pearson,\n"
                    "RT(ead) tanimoto,\n"
                    "WT(rite) tanimoto,\n"
                    "RJ(ead) jaccard,\n"
                    "WJ(rite) jaccard,\n"
                    "RC(ead) cosine,\n"
                    "WC(rite) cosine,\n"
                    "RS(ead) spearman,\n"
                    "WS(rite) spearman,\n"
                    "RK(ead) kendall tau, or\n"
                    "WK(rite) kendall tau? ==> "
                )

                n_neighbors = int(input("Number of neighbors: "))
                sim_sig_weighting = int(input("Significance weighting cutoff: "))

                try:
                    if sub_cmd == "RD" or sub_cmd == "rd":
                        # Load the dictionary back from the pickle file.
                        itemsim = pickle.load(
                            open(
                                f"sim_mat/save_itemsim_distance_{dataset_name}.p", "rb"
                            )
                        )
                        sim_method = "sim_distance"
                        sim_ran = True

                    elif sub_cmd == "WD" or sub_cmd == "wd":
                        # transpose the U-I matrix and calc item-item similarities matrix
                        itemsim = calculateSimilarItems(
                            prefs,
                            neighbors=n_neighbors,
                            similarity=sim_distance,
                            sim_sig_weighting=sim_sig_weighting,
                        )
                        # Dump/save dictionary to a pickle file
                        pickle.dump(
                            itemsim,
                            open(
                                f"sim_mat/save_itemsim_distance_{dataset_name}.p", "wb"
                            ),
                        )
                        sim_method = "sim_distance"
                        sim_ran = True

                    elif sub_cmd == "RP" or sub_cmd == "rp":
                        # Load the dictionary back from the pickle file.
                        itemsim = pickle.load(
                            open(f"sim_mat/save_itemsim_pearson_{dataset_name}.p", "rb")
                        )
                        sim_method = "sim_pearson"
                        sim_ran = True

                    elif sub_cmd == "WP" or sub_cmd == "wp":
                        # transpose the U-I matrix and calc item-item similarities matrix
                        itemsim = calculateSimilarItems(
                            prefs,
                            neighbors=n_neighbors,
                            similarity=sim_pearson,
                            sim_sig_weighting=sim_sig_weighting,
                        )
                        # Dump/save dictionary to a pickle file
                        pickle.dump(
                            itemsim,
                            open(
                                f"sim_mat/save_itemsim_pearson_{dataset_name}.p", "wb"
                            ),
                        )
                        sim_method = "sim_pearson"
                        sim_ran = True

                    elif sub_cmd == "RT" or sub_cmd == "rt":
                        # Load the dictionary back from the pickle file.
                        itemsim = pickle.load(
                            open(
                                f"sim_mat/save_itemsim_tanimoto_{dataset_name}.p", "rb"
                            )
                        )
                        sim_method = "sim_tanimoto"
                        sim_ran = True

                    elif sub_cmd == "WT" or sub_cmd == "wt":
                        # transpose the U-I matrix and calc item-item similarities matrix
                        itemsim = calculateSimilarItems(
                            prefs,
                            neighbors=n_neighbors,
                            similarity=sim_tanimoto,
                            sim_sig_weighting=sim_sig_weighting,
                        )
                        # Dump/save dictionary to a pickle file
                        pickle.dump(
                            itemsim,
                            open(
                                f"sim_mat/save_itemsim_tanimoto_{dataset_name}.p", "wb"
                            ),
                        )
                        sim_method = "sim_tanimoto"
                        sim_ran = True

                    elif sub_cmd == "RJ" or sub_cmd == "rj":
                        # Load the dictionary back from the pickle file.
                        itemsim = pickle.load(
                            open(f"sim_mat/save_itemsim_jaccard_{dataset_name}.p", "rb")
                        )
                        sim_method = "sim_jaccard"
                        sim_ran = True

                    elif sub_cmd == "WJ" or sub_cmd == "wj":
                        # transpose the U-I matrix and calc item-item similarities matrix
                        itemsim = calculateSimilarItems(
                            prefs,
                            neighbors=n_neighbors,
                            similarity=sim_jaccard,
                            sim_sig_weighting=sim_sig_weighting,
                        )
                        # Dump/save dictionary to a pickle file
                        pickle.dump(
                            itemsim,
                            open(
                                f"sim_mat/save_itemsim_jaccard_{dataset_name}.p", "wb"
                            ),
                        )
                        sim_method = "sim_jaccard"
                        sim_ran = True

                    elif sub_cmd == "RC" or sub_cmd == "rc":
                        # Load the dictionary back from the pickle file.
                        itemsim = pickle.load(
                            open(f"sim_mat/save_itemsim_cosine_{dataset_name}.p", "rb")
                        )
                        sim_method = "sim_cosine"
                        sim_ran = True

                    elif sub_cmd == "WC" or sub_cmd == "wc":
                        # transpose the U-I matrix and calc item-item similarities matrix
                        itemsim = calculateSimilarItems(
                            prefs,
                            neighbors=n_neighbors,
                            similarity=sim_cosine,
                            sim_sig_weighting=sim_sig_weighting,
                        )
                        # Dump/save dictionary to a pickle file
                        pickle.dump(
                            itemsim,
                            open(f"sim_mat/save_itemsim_cosine_{dataset_name}.p", "wb"),
                        )
                        sim_method = "sim_cosine"
                        sim_ran = True

                    elif sub_cmd == "RS" or sub_cmd == "rs":
                        # Load the dictionary back from the pickle file.
                        itemsim = pickle.load(
                            open(
                                f"sim_mat/save_itemsim_spearman_{dataset_name}.p", "rb"
                            )
                        )
                        sim_method = "sim_spearman"
                        sim_ran = True

                    elif sub_cmd == "WS" or sub_cmd == "ws":
                        # transpose the U-I matrix and calc item-item similarities matrix
                        itemsim = calculateSimilarItems(
                            prefs,
                            neighbors=n_neighbors,
                            similarity=sim_spearman,
                            sim_sig_weighting=sim_sig_weighting,
                        )
                        # Dump/save dictionary to a pickle file
                        pickle.dump(
                            itemsim,
                            open(
                                f"sim_mat/save_itemsim_spearman_{dataset_name}.p", "wb"
                            ),
                        )
                        sim_method = "sim_spearman"
                        sim_ran = True

                    elif sub_cmd == "RK" or sub_cmd == "rk":
                        # Load the dictionary back from the pickle file.
                        itemsim = pickle.load(
                            open(
                                f"sim_mat/save_itemsim_kendall_tau_{dataset_name}.p",
                                "rb",
                            )
                        )
                        sim_method = "sim_kendall_tau"
                        sim_ran = True

                    elif sub_cmd == "WK" or sub_cmd == "wk":
                        # transpose the U-I matrix and calc item-item similarities matrix
                        itemsim = calculateSimilarItems(
                            prefs,
                            neighbors=n_neighbors,
                            similarity=sim_kendall_tau,
                            sim_sig_weighting=sim_sig_weighting,
                        )
                        # Dump/save dictionary to a pickle file
                        pickle.dump(
                            itemsim,
                            open(
                                f"sim_mat/save_itemsim_kendall_tau_{dataset_name}.p",
                                "wb",
                            ),
                        )
                        sim_method = "sim_kendall_tau"
                        sim_ran = True

                    else:
                        print("Sim sub-command %s is invalid, try again" % sub_cmd)
                        continue

                    ready = True  # sub command completed successfully

                except Exception as ex:
                    print(
                        "Error!!",
                        ex,
                        "\nNeed to W(rite) a file before you can R(ead) it!"
                        " Enter Sim(ilarity matrix) again and choose a Write command",
                    )
                    print()

                if len(itemsim) > 0 and ready == True:
                    # Only want to print if sub command completed successfully
                    print(
                        "Similarity matrix based on %s, len = %d"
                        % (sim_method, len(itemsim))
                    )
                    print()

                    items = list(itemsim.keys())
                    for i, item in enumerate(items):
                        if i >= 10:
                            break
                        print(
                            f"itemsim entry {i+1}: {item}, len(entry): {len(itemsim[item])}"
                        )
                        print(itemsim[item][:5])

                print()

            else:
                print("Empty dictionary, R(ead) in some data!")

        elif file_io == "Simu" or file_io == "simu":
            print()
            if len(prefs) > 0:
                usersim = {}
                ready = False  # sub command in progress
                sub_cmd = input(
                    "RD(ead) distance,\n"
                    "WD(rite) distance,\n"
                    "RP(ead) pearson,\n"
                    "WP(rite) pearson,\n"
                    "RT(ead) tanimoto,\n"
                    "WT(rite) tanimoto,\n"
                    "RJ(ead) jaccard,\n"
                    "WJ(rite) jaccard,\n"
                    "RC(ead) cosine,\n"
                    "WC(rite) cosine,\n"
                    "RS(ead) spearman,\n"
                    "WS(rite) spearman,\n"
                    "RK(ead) kendall tau, or\n"
                    "WK(rite) kendall tau? ==> "
                )

                n_neighbors = int(input("Number of neighbors: "))
                sim_sig_weighting = int(input("Similarity significance weighting: "))
                
                try:
                    if sub_cmd == "RD" or sub_cmd == "rd":
                        # Load the dictionary back from the pickle file.
                        usersim = pickle.load(
                            open(
                                f"sim_mat/save_usersim_distance_{dataset_name}.p", "rb"
                            )
                        )
                        sim_method = "sim_distance"

                    elif sub_cmd == "WD" or sub_cmd == "wd":
                        # transpose the U-I matrix and calc item-item similarities matrix
                        usersim = calculateSimilarUsers(
                            prefs,
                            neighbors=n_neighbors,
                            similarity=sim_distance,
                            sim_sig_weighting=sim_sig_weighting,
                        )
                        # Dump/save dictionary to a pickle file
                        pickle.dump(
                            usersim,
                            open(
                                f"sim_mat/save_usersim_distance_{dataset_name}.p", "wb"
                            ),
                        )
                        sim_method = "sim_distance"

                    elif sub_cmd == "RP" or sub_cmd == "rp":
                        # Load the dictionary back from the pickle file.
                        usersim = pickle.load(
                            open(f"sim_mat/save_usersim_pearson_{dataset_name}.p", "rb")
                        )
                        sim_method = "sim_pearson"

                    elif sub_cmd == "WP" or sub_cmd == "wp":
                        # transpose the U-I matrix and calc item-item similarities matrix
                        usersim = calculateSimilarUsers(
                            prefs,
                            neighbors=n_neighbors,
                            similarity=sim_pearson,
                            sim_sig_weighting=sim_sig_weighting,
                        )
                        # Dump/save dictionary to a pickle file
                        pickle.dump(
                            usersim,
                            open(
                                f"sim_mat/save_usersim_pearson_{dataset_name}.p", "wb"
                            ),
                        )
                        sim_method = "sim_pearson"

                    elif sub_cmd == "RT" or sub_cmd == "rt":
                        # Load the dictionary back from the pickle file.
                        usersim = pickle.load(
                            open(
                                f"sim_mat/save_usersim_tanimoto_{dataset_name}.p", "rb"
                            )
                        )
                        sim_method = "sim_tanimoto"

                    elif sub_cmd == "WT" or sub_cmd == "wt":
                        # transpose the U-I matrix and calc item-item similarities matrix
                        usersim = calculateSimilarUsers(
                            prefs,
                            neighbors=n_neighbors,
                            similarity=sim_tanimoto,
                            sim_sig_weighting=sim_sig_weighting,
                        )
                        # Dump/save dictionary to a pickle file
                        pickle.dump(
                            usersim,
                            open(
                                f"sim_mat/save_usersim_tanimoto_{dataset_name}.p", "wb"
                            ),
                        )
                        sim_method = "sim_tanimoto"

                    elif sub_cmd == "RJ" or sub_cmd == "rj":
                        # Load the dictionary back from the pickle file.
                        usersim = pickle.load(
                            open(f"sim_mat/save_usersim_jaccard_{dataset_name}.p", "rb")
                        )
                        sim_method = "sim_jaccard"

                    elif sub_cmd == "WJ" or sub_cmd == "wj":
                        # transpose the U-I matrix and calc item-item similarities matrix
                        usersim = calculateSimilarUsers(
                            prefs,
                            neighbors=n_neighbors,
                            similarity=sim_jaccard,
                            sim_sig_weighting=sim_sig_weighting,
                        )
                        # Dump/save dictionary to a pickle file
                        pickle.dump(
                            usersim,
                            open(
                                f"sim_mat/save_usersim_jaccard_{dataset_name}.p", "wb"
                            ),
                        )
                        sim_method = "sim_jaccard"

                    elif sub_cmd == "RC" or sub_cmd == "rc":
                        # Load the dictionary back from the pickle file.
                        usersim = pickle.load(
                            open(f"sim_mat/save_usersim_cosine_{dataset_name}.p", "rb")
                        )
                        sim_method = "sim_cosine"

                    elif sub_cmd == "WC" or sub_cmd == "wc":
                        # transpose the U-I matrix and calc item-item similarities matrix
                        usersim = calculateSimilarUsers(
                            prefs,
                            neighbors=n_neighbors,
                            similarity=sim_cosine,
                            sim_sig_weighting=sim_sig_weighting,
                        )
                        # Dump/save dictionary to a pickle file
                        pickle.dump(
                            usersim,
                            open(f"sim_mat/save_usersim_cosine_{dataset_name}.p", "wb"),
                        )
                        sim_method = "sim_cosine"

                    elif sub_cmd == "RS" or sub_cmd == "rs":
                        # Load the dictionary back from the pickle file.
                        usersim = pickle.load(
                            open(
                                f"sim_mat/save_usersim_spearman_{dataset_name}.p", "rb"
                            )
                        )
                        sim_method = "sim_spearman"

                    elif sub_cmd == "WS" or sub_cmd == "ws":
                        # transpose the U-I matrix and calc item-item similarities matrix
                        usersim = calculateSimilarUsers(
                            prefs,
                            neighbors=n_neighbors,
                            similarity=sim_spearman,
                            sim_sig_weighting=sim_sig_weighting,
                        )
                        # Dump/save dictionary to a pickle file
                        pickle.dump(
                            usersim,
                            open(
                                f"sim_mat/save_usersim_spearman_{dataset_name}.p", "wb"
                            ),
                        )
                        sim_method = "sim_spearman"

                    elif sub_cmd == "RK" or sub_cmd == "rk":
                        # Load the dictionary back from the pickle file.
                        usersim = pickle.load(
                            open(
                                f"sim_mat/save_usersim_kendall_tau_{dataset_name}.p",
                                "rb",
                            )
                        )
                        sim_method = "sim_kendall_tau"

                    elif sub_cmd == "WK" or sub_cmd == "wk":
                        # transpose the U-I matrix and calc item-item similarities matrix
                        usersim = calculateSimilarUsers(
                            prefs,
                            neighbors=n_neighbors,
                            similarity=sim_kendall_tau,
                            sim_sig_weighting=sim_sig_weighting,
                        )
                        # Dump/save dictionary to a pickle file
                        pickle.dump(
                            usersim,
                            open(
                                f"sim_mat/save_usersim_kendall_tau_{dataset_name}.p",
                                "wb",
                            ),
                        )
                        sim_method = "sim_kendall_tau"

                    else:
                        print("Sim sub-command %s is invalid, try again" % sub_cmd)
                        continue

                    ready = True  # sub command completed successfully

                except Exception as ex:
                    print(
                        "Error!!",
                        ex,
                        "\nNeed to W(rite) a file before you can R(ead) it!"
                        " Enter Simu(ser similarity matrix) again and choose a Write command",
                    )
                    print()

                if len(usersim) > 0 and ready == True:
                    # Only want to print if sub command completed successfully
                    print(
                        "Similarity matrix based on %s, len = %d"
                        % (sim_method, len(usersim))
                    )
                    print()

                    users = list(usersim.keys())
                    for i, user in enumerate(users):
                        if i >= 50:
                            break
                        print(
                            f"User {user} has (partial) usersim matrix: {usersim[user][:5]}"
                        )

                print()

            else:
                print("Empty dictionary, R(ead) in some data!")
       
        elif file_io == 'T' or file_io == 't':
            if len(prefs) > 0:
                answer = input('Generate both test and train data? Y or y, N or n: ')
                if answer == 'N' or answer == 'n':
                    TRAIN_ONLY = True
                else:
                    TRAIN_ONLY = False
                
                #print('TRAIN_ONLY  in EVAL =', TRAIN_ONLY) ## debug
                train, test = train_test_split(ratings, TRAIN_ONLY) ## this should 
                ##     be only place where TRAIN_ONLY is needed!! 
                ##     Check for len(test)==0 elsewhere
                
                test_train_info(test, train) ## print test/train info
        
                ## How is MSE calculated for train?? self.ratings is the train
                ##    data when ExplicitMF is instantiated for both als and sgd.
                ##    So, MSE calc is between predictions based on train data 
                ##    against the actuals for train data
                ## How is MSE calculated for test?? It's comparing the predictions
                ##    based on train data against the actuals for test data
                
                test_train_done = True
                print()
                print('Test and Train arrays are ready!')
                print()
            else:
                print ('Empty U-I matrix, read in some data!')
                print() 

        elif file_io == 'MF-ALS' or file_io == 'mf-als':
            
            if len(ratings) > 0:
                if test_train_done:
                    
                    ## als processing
                    
                    print()
                    ## sample instantiations ..
                    if len(ratings) < 10: ## for critics
                        print('Sample for critics .. ')
                        iter_array = [1, 2, 5, 10, 20]
                        MF_ALS = ExplicitMF(train, learning='als', n_factors=2, user_fact_reg=1, item_fact_reg=1, max_iters=max(iter_array), verbose=True)
                        print('[2,1,20]')
                    
                    elif len(ratings) < 1000: ## for ml-100k
                        print('Sample for ml-100k .. ')
                        iter_array = [1, 2, 5 , 10, 20, 50] #, 100] #, 200]
                        MF_ALS = ExplicitMF(train, learning='als', n_factors=20, user_fact_reg=.01, item_fact_reg=.01, max_iters=max(iter_array), verbose=True) 
                        print('[20,0.01,50]')
                    
                    elif len(ratings) < 10000: ## for ml-1m
                        print('Sample for ml-1M .. ')
                        iter_array = [1, 2, 5, 10] 
                        MF_ALS = ExplicitMF(train, learning='als', n_factors=20, user_fact_reg=.1, item_fact_reg=.1, max_iters=max(iter_array), verbose=True) 
                        
                    parms = input('Y or y to use these parameters or Enter to modify: ')# [2,0.01,10,False]
                    if parms == 'Y' or parms == 'y':
                        pass
                    else:
                        parms = eval(input('Enter new parameters as a list: [n_factors, reg, iters]: '))
                        
                        # instantiate with this set of parms
                        MF_ALS = ExplicitMF(train,learning='als', 
                                            n_factors=parms[0], 
                                            user_fact_reg=parms[1], 
                                            item_fact_reg=parms[1])
                       
                        # set up the iter_array for this run to pass on
                        orig_iter_array = [1, 2, 5, 10, 20, 50, 100, 200]
                        i_max = parms[2]
                        index = orig_iter_array.index(i_max)
                        iter_array = []
                        for i in range(0, index+1):
                            iter_array.append(orig_iter_array[i])
                            
                    # run the algo and plot results
                    MF_ALS.calculate_learning_curve(iter_array, test) 
                    plot_learning_curve(iter_array, MF_ALS, parms, dataset)
                    print(parms[0])
                    np.save(f'results_als_{dataset}/ALS_{dataset}_{parms[0]}_{parms[1]}_{parms[2]}_train.npy', MF_ALS.train_mse)
                    np.save(f'results_als_{dataset}/ALS_{dataset}_{parms[0]}_{parms[1]}_{parms[2]}_test.npy', MF_ALS.test_mse)
                    
                    with open(f"results_als_{dataset}/ALS_{dataset}_train_mse_grid_search_results.csv", "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([f'ALS~{parms[0]}~{parms[1]}'] + MF_ALS.train_mse)

                    with open(f"results_als_{dataset}/ALS_{dataset}_test_mse_grid_search_results.csv", "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([f'ALS~{parms[0]}~{parms[1]}'] + MF_ALS.test_mse)

                else:
                    print ('Empty test/train arrays, run the T command!')
                    print()                    
            else:
                print ('Empty U-I matrix, read in some data!')
                print()
                    
        elif file_io == 'MF-SGD' or file_io == 'mf-sgd':
            
            if len(ratings) > 0:
                
                if test_train_done:
                
                    ## sgd processing
                     
                    ## sample instantiations ..
                    if len(ratings) < 10: ## for critics
                        # Use these parameters for small matrices
                        print('Sample for critics .. ')
                        iter_array = [1, 2, 5, 10, 20]                     
                        MF_SGD = ExplicitMF(train, 
                                            n_factors=2, 
                                            learning='sgd', 
                                            sgd_alpha=0.075,
                                            sgd_beta=0.01, 
                                            max_iters=max(iter_array), 
                                            sgd_random=False)

                    elif len(ratings) < 1000:
                       # Use these parameters for ml-100k
                        print('Sample for ml-100k .. ')
                        iter_array = [1, 2, 5, 10, 20]                     
                        MF_SGD = ExplicitMF(train, 
                                            n_factors=2, 
                                            learning='sgd', 
                                            sgd_alpha=0.02,
                                            sgd_beta=0.2, 
                                            max_iters=max(iter_array), 
                                            sgd_random=False, verbose=True)
                        

                         
                    elif len(ratings) < 10000:
                       # Use these parameters for ml-1m
                        print('Sample for ml-1m .. ')
                        iter_array = [1, 2, 5, 10] #, 20, 50, 100]                     
                        MF_SGD = ExplicitMF(train, 
                                            n_factors=20, 
                                            learning='sgd', 
                                            sgd_alpha=0.1,
                                            sgd_beta=0.1, 
                                            max_iters=max(iter_array), 
                                            sgd_random=False, verbose=True)
                        
                    parms = input('Y or y to use these parameters or Enter to modify: ')# [2,0.01,10,False]
                    if parms == 'Y' or parms == 'y':
                        pass
                    else:
                        parms = eval(input('Enter new parameters as a list: [n_factors K, learning_rate alpha, reg beta, max_iters: ')) #', random]: '))
                        MF_SGD = ExplicitMF(train, n_factors=parms[0], 
                                            learning='sgd', 
                                            sgd_alpha=parms[1], 
                                            sgd_beta=parms[2], 
                                            max_iters=parms[3], 
                                            sgd_random=False, verbose=True)  

                        orig_iter_array = [1, 2, 5, 10, 20, 50, 100, 200]
                        i_max = parms[3]
                        index = orig_iter_array.index(i_max)
                        iter_array = []
                        for i in range(0, index+1):
                            iter_array.append(orig_iter_array[i])
                         
                    MF_SGD.calculate_learning_curve(iter_array, test) # start the training
                    plot_learning_curve(iter_array, MF_SGD, parms, dataset)    

                    np.save(f'results_sgd_{dataset}/SGD_{dataset}_{parms[0]}_{parms[1]}_{parms[2]}_{parms[3]}_train.npy', MF_SGD.train_mse)
                    np.save(f'results_sgd_{dataset}/SGD_{dataset}_{parms[0]}_{parms[1]}_{parms[2]}_{parms[3]}_test.npy', MF_SGD.test_mse)
                    with open(f"results_sgd_{dataset}/SGD_{dataset}_train_mse_grid_search_results.csv", "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([f'SGD~{parms[0]}~{parms[1]}~{parms[2]}'] + MF_SGD.train_mse)

                    with open(f"results_sgd_{dataset}/SGD_{dataset}_test_mse_grid_search_results.csv", "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([f'SGD~{parms[0]}~{parms[1]}~{parms[2]}'] + MF_SGD.test_mse)

                else:
                    print ('Empty test/train arrays, run the T command!')
                    print()   

        elif file_io == 'MF-SGD-GRID' or file_io == 'mf-sgd-grid':

            k_list = [2,20,200]
            alpha_list = [0.02,0.002,0.0002]
            beta_list = [0.2,0.02,0.002]
            iters = 20

            train_mse_list = []
            test_mse_list = []
            for k in k_list:
                for alpha in alpha_list:
                    for beta in beta_list:
                        parms = [k, alpha, beta, iters]
                        MF_SGD = ExplicitMF(train, n_factors=k, 
                                                    learning='sgd', 
                                                    sgd_alpha=alpha, 
                                                    sgd_beta=beta, 
                                                    max_iters=iters, 
                                                    sgd_random=False, verbose=True)  

                        orig_iter_array = [1, 2, 5, 10, 20, 50, 100, 200]
                        i_max = iters
                        index = orig_iter_array.index(i_max)
                        iter_array = []
                        for i in range(0, index+1):
                            iter_array.append(orig_iter_array[i])
                                
                        MF_SGD.calculate_learning_curve(iter_array, test) # start the training
                        plot_learning_curve(iter_array, MF_SGD, parms, dataset)    

                        print([f'SGD~{k}~{alpha}~{beta}'] + MF_SGD.train_mse)
                        train_mse_list.append([f'SGD~{k}~{alpha}~{beta}'] + MF_SGD.train_mse)
                        print([f'SGD~{k}~{alpha}~{beta}'] + MF_SGD.test_mse)
                        test_mse_list.append([f'SGD~{k}~{alpha}~{beta}'] + MF_SGD.test_mse)
                        print()

            with open(f"results_sgd_{dataset}/SGD_{dataset}_train_mse_grid_search_results.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(train_mse_list)

            with open(f"results_sgd_{dataset}/SGD_{dataset}_test_mse_grid_search_results.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(test_mse_list)
        
        elif file_io == 'MF-ALS-GRID' or file_io == 'mf-als-grid':

            f_list = [2,20,100,200]
            regLambda_list = [1,0.1,0.01,0.001,0.0001,0.00001]         
            iters = 20  

            train_mse_list = []
            test_mse_list = []
            for f in f_list:
                for regLambda in regLambda_list:
                    parms = [f,regLambda,iters]
                    # instantiate with this set of parms
                    MF_ALS = ExplicitMF(train,learning='als', 
                                        n_factors=f, 
                                        user_fact_reg=regLambda, 
                                        item_fact_reg=regLambda)
                    
                    # set up the iter_array for this run to pass on
                    orig_iter_array = [1, 2, 5, 10, 20, 50, 100, 200]
                    i_max = iters
                    index = orig_iter_array.index(i_max)
                    iter_array = []
                    for i in range(0, index+1):
                        iter_array.append(orig_iter_array[i])
                                
                    MF_ALS.calculate_learning_curve(iter_array, test) 
                    plot_learning_curve(iter_array, MF_ALS, parms, dataset)
                    print([f'ALS~{f}~{regLambda}'] + MF_ALS.train_mse)
                    train_mse_list.append([f'ALS~{f}~{regLambda}'] + MF_ALS.train_mse)
                    print([f'ALS~{f}~{regLambda}'] + MF_ALS.test_mse)
                    test_mse_list.append([f'ALS~{f}~{regLambda}'] + MF_ALS.test_mse)
            
            with open(f"results_als_{dataset}/ALS_{dataset}_train_mse_grid_search_results.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(train_mse_list)
            
            with open(f"results_als_{dataset}/ALS_{dataset}_test_mse_grid_search_results.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(test_mse_list)

        elif file_io == "I" or file_io == "i":
            print()
            if len(prefs) > 0 and len(itemsim) > 0:
                print("Example:")
                user_name = "Toby"

                print(
                    "Item-based CF recs for %s, %s: " % (user_name, sim_method),
                    getRecommendedItems(prefs, itemsim, user_name, threshold, sim_sig_weighting),
                )

                print()

                print("Item-based CF recommendations for all users:")
                # Calc Item-based CF recommendations for all users

                get_all_II_recs(prefs, itemsim, sim_method)

                print()

            else:
                if len(prefs) == 0:
                    print("Empty dictionary, R(ead) in some data!")
                else:
                    print(
                        "Empty similarity matrix, use Sim(ilarity) to create a sim matrix!"
                    )

        elif file_io == "LCVSIM" or file_io == "lcvsim":
            print()
            try:
                if len(prefs) > 0:
                    print("LOO_CV_SIM Evaluation")

                    input_algo = input(
                        "UU-CF, II-CF, TFIDF, or hybrid recommendations? "
                    )

                    if sim_method == 'sim_distance':
                        sim = sim_distance
                    elif sim_method == 'sim_pearson':
                        sim = sim_pearson

                    if input_algo.lower() == "uu-cf":
                        threshold = float(input("Similarity threshold: "))
                        sim_sig_weighting = float(input('Similarity significance weighting: '))
                        algo = new_getRecommendedUsers
                        sim_matrix = usersim
                        mov = None
                    elif input_algo.lower() == "ii-cf":
                        threshold = float(input("Similarity threshold: "))
                        sim_sig_weighting = float(input('Similarity significance weighting: '))
                        algo = new_getRecommendedItems  # Item-based recommendation
                        sim_matrix = itemsim
                        mov = None
                    elif input_algo.lower() == "tfidf":
                        threshold = float(input("Similarity threshold: "))
                        sim_sig_weighting = 0 # TFIDF doesn't use a sim sig weighting
                        n_neighbors = 0 # TFIDF doesn't use n_neighbors
                        algo = new_get_TFIDF_recommendations
                        sim_matrix = cosim_matrix
                        sim = 'TFIDF'
                        mov = movies

                        error_total, error_list = loo_cv_sim(
                            prefs,
                            sim,
                            algo,
                            sim_matrix,
                            dataset_name,
                            threshold,
                            sim_sig_weighting,
                            n_neighbors,
                            mov,
                        )
                    elif input_algo.lower() == "hybrid":
                        algo = new_get_hybrid_recommendations
                        mov = movies
                        threshold = float(input("Similarity threshold: "))

                        try:
                            sim_matrix = updated_cosim_matrix
                        except:
                            print("Please run the HYB command first.")
                            return

                        error_total, error_list = loo_cv_sim(
                            prefs,
                            sim,
                            algo,
                            sim_matrix,
                            dataset_name,
                            threshold,
                            sim_sig_weighting,
                            n_neighbors,
                            mov,
                            weighting_factor
                        )
                        
                    else:
                        print(
                            'Invalid recommendation algo. Please say "user" or "item".'
                        )
                        return

                    if not sim_ran:
                        continue
                        print()

                    elif sim_method == "sim_pearson":
                        sim = sim_pearson
                        error_total, error_list = loo_cv_sim(
                            prefs,
                            sim,
                            algo,
                            sim_matrix,
                            dataset_name,
                            threshold,
                            sim_sig_weighting,
                            n_neighbors,
                            mov,
                        )

                        print()

                    elif sim_method == "sim_distance":
                        sim = sim_distance
                        error_total, error_list = loo_cv_sim(
                            prefs,
                            sim,
                            algo,
                            sim_matrix,
                            dataset_name,
                            threshold,
                            sim_sig_weighting,
                            n_neighbors,
                            mov,
                        )

                        print()

                    elif sim_method == "sim_tanimoto":
                        sim = sim_tanimoto
                        error_total, error_list = loo_cv_sim(
                            prefs,
                            sim,
                            algo,
                            sim_matrix,
                            dataset_name,
                            threshold,
                            sim_sig_weighting,
                            n_neighbors,
                            mov,
                        )

                        print()

                    elif sim_method == "sim_jaccard":
                        sim = sim_jaccard
                        error_total, error_list = loo_cv_sim(
                            prefs,
                            sim,
                            algo,
                            sim_matrix,
                            dataset_name,
                            threshold,
                            sim_sig_weighting,
                            n_neighbors,
                            mov,
                        )

                        print()

                    elif sim_method == "sim_cosine":
                        sim = sim_cosine
                        error_total, error_list = loo_cv_sim(
                            prefs,
                            sim,
                            algo,
                            sim_matrix,
                            dataset_name,
                            threshold,
                            sim_sig_weighting,
                            n_neighbors,
                            mov,
                        )

                        print()

                    elif sim_method == "sim_spearman":
                        sim = sim_spearman
                        error_total, error_list = loo_cv_sim(
                            prefs,
                            sim,
                            algo,
                            sim_matrix,
                            dataset_name,
                            threshold,
                            sim_sig_weighting,
                            n_neighbors,
                            mov,
                        )

                        print()

                    elif sim_method == "sim_kendall_tau":
                        sim = sim_kendall_tau
                        error_total, error_list = loo_cv_sim(
                            prefs,
                            sim,
                            algo,
                            sim_matrix,
                            dataset_name,
                            threshold,
                            sim_sig_weighting,
                            n_neighbors,
                            mov,
                        )

                        print()

                    else:
                        print(
                            "Run Sim(ilarity matrix) command to create/load Sim matrix!"
                        )
                    if dataset_name == "critics":
                        print(error_list)

                    # # for anova testing purposes
                    # sim_str = str(sim).split()[1]
                    # algo_str = str(algo).split()[1]
                    # pickle.dump(
                    #     error_list,
                    #     open(
                    #         f"errors/sq_errors_{dataset_name}_{str(threshold).replace('.',',')}_{sim_sig_weighting}_{sim_str}_{algo_str}_{n_neighbors}.p",
                    #         "wb",
                    #     ),
                    # )

                else:
                    print("Empty dictionary, run R(ead) OR Empty Sim Matrix, run Sim!")
            except Exception as ex:
                print(f"{ex}: Empty Sim Matrix, run Sim!")
                tb = traceback.format_exc()
                print(str(tb))

        elif file_io == "TFIDF" or file_io == "tfidf":
            print()
            # determine the U-I matrix to use ..
            if len(prefs) > 0 and len(prefs) <= 10:  # critics
                # convert prefs dictionary into 2D list
                R = to_array(prefs)
                feature_str = to_string(features)
                feature_docs = to_docs(feature_str, genres)

                print("critics")
                print(R)
                print()
                print("features")
                print(features)
                print()
                print("feature docs")
                print(feature_docs)
                cosim_matrix = cosine_sim(feature_docs)
                print()
                print("cosine sim matrix")
                print(cosim_matrix)

                sim = 'TFIDF'
                sim_method = 'TFIDF'
                tfidf_ran = True

            elif len(prefs) > 10:
                print("ml-100k")
                # convert prefs dictionary into 2D list
                R = to_array(prefs)
                feature_str = to_string(features)
                feature_docs = to_docs(feature_str, genres)

                print(R[:3][:5])
                print()
                print("features")
                print(features[0:5])
                print()
                print("feature docs")
                print(feature_docs[0:5])
                cosim_matrix = cosine_sim(feature_docs)
                print()
                print("cosine sim matrix")
                print(type(cosim_matrix), len(cosim_matrix))
                print(cosim_matrix)
                print()

                cosim_matrix_bl = np.multiply(
                    cosim_matrix, np.tri(cosim_matrix.shape[0], k=-1)
                )
                print("cosine sim matrix bottom left only")
                print(cosim_matrix_bl)
                print()
                # plt.hist(cosim_matrix_bl)
                # plt.savefig('figures/cosim_matrix_with_0.png')
                # plt.close()

                num_zeros = len(cosim_matrix_bl[cosim_matrix_bl == 0])
                print("num zeros")
                print(num_zeros)

                cosim_matrix_bl_wo_zeros = cosim_matrix_bl[cosim_matrix_bl != 0]
                print("num non-zero values")
                print(len(cosim_matrix_bl_wo_zeros))
                print()
                # plt.hist(cosim_matrix_bl_wo_zeros)
                # plt.vlines(x = 0.306, ymin = 0, ymax = 120000, color = 'r')
                # plt.vlines(x = 0.577, ymin = 0, ymax = 120000, color = 'r')
                # plt.vlines(x = 0.847, ymin = 0, ymax = 120000, color = 'r')
                # plt.savefig('figures/cosim_matrix_without_0.png')
                # plt.close()

                cosim_matrix_mean = np.mean(cosim_matrix_bl_wo_zeros)
                cosim_matrix_std = np.std(cosim_matrix_bl_wo_zeros)
                print("mean of cosim matrix")
                print(cosim_matrix_mean)
                print("std of cosim matrix")
                print(cosim_matrix_std)
                print()
                
                sim = 'TFIDF'
                sim_method = 'TFIDF'
                tfidf_ran = True
            else:
                print("Empty dictionary, read in some data!")
                print()

        elif file_io == "HYB" or file_io == "hyb":

            weighting_factors = [0, 0.25, 0.5, 0.75, 1]
            weighting_factor = float(
                input("Weighting factor (0, 0.25, 0.5, 0.75, or 1): ")
            )
            if weighting_factor in weighting_factors:
                itemsim_mat = itemsim_to_np_matrix(itemsim, movies)
                updated_cosim_matrix = hybrid_update_sim_matrix(
                    cosim_matrix, itemsim_mat, weighting_factor
                )
                hybrid_ran = True
            else:
                print("Input a valid weighting factor")

        elif file_io == 'TTV' or file_io == 'ttv':
            if pd_rml_ran == True or pd_r_ran == True:
                print('Creating train, test, and validation splits...')
                train, test, val = train_test_validation_split(df)
                print('Finished creating train, test, and validation splits!')

                ttv_ran = True
            else:
                print('Run PD-RML or PD-R first!')
        
        elif file_io == 'BNCF' or file_io == 'bncf':
            if pd_r_ran == True or pd_rml_ran == True:

                n_factors = int(input('Number of Factors: '))
                lr = float(input('Learning rate: '))
                dropout_prob = float(input('Dropout probability (default = 0.2): '))
                n_nodes_per_layer_list = input('Number of nodes per layer (ie. [64, 32, 16, 8, 4, 2]): ')
                n_nodes_per_layer_list = n_nodes_per_layer_list.strip('][').split(', ')
                n_nodes_per_layer_list = [int(i) for i in n_nodes_per_layer_list]
                # creating item embedding path
                movie_input = Input(shape=[1], name="Item-Input")
                movie_embedding = Embedding(n_items+1, n_factors, name="Item-Embedding")(movie_input)
                movie_vec = Flatten(name="Flatten-Items")(movie_embedding)

                # creating user embedding path
                user_input = Input(shape=[1], name="User-Input")
                user_embedding = Embedding(n_users+1, n_factors, name="User-Embedding")(user_input)
                user_vec = Flatten(name="Flatten-Users")(user_embedding)

                # concatenate features
                conc = Concatenate()([movie_vec, user_vec])

                # add fully-connected-layers
                dense = Dense(n_nodes_per_layer_list[0], activation='relu')(conc)
                dropout = Dropout(dropout_prob)(dense)
                batch_norm = BatchNormalization()(dropout)

                for k, n_nodes in enumerate(n_nodes_per_layer_list[1:-1]):
                    dense = Dense(n_nodes, activation='relu')(batch_norm)
                    dropout = Dropout(dropout_prob)(dense)
                    batch_norm = BatchNormalization()(dropout)

                dense = Dense(n_nodes_per_layer_list[-1], activation='relu')(batch_norm)
                out = Dense(1)(dense)

                # Create model and compile it
                model = Model([user_input, movie_input], out)
                model.compile(optimizer=Adam(learning_rate=lr), loss=MeanSquaredError())
                model.summary()
            
                ncf_built = True
            else: 
                print('Run PD-RML or PD-R first!')

        elif file_io == 'TNCF' or file_io == 'tncf':
            if ncf_built == True and ttv_ran == True:
                epochs = int(input('Number of Epochs: ')) # default 250
                batch_size = int(input('Batch size: '))
                patience = int(input('Patience: '))
                early_stopping_metric = 'val_loss'

                callback = EarlyStopping(monitor=early_stopping_metric, patience=patience)
                history = model.fit(x = [train.user_id, train.item_id], 
                                    y = train.rating, 
                                    validation_data = ((val.user_id, val.item_id), val.rating), 
                                    epochs=epochs, 
                                    verbose=1, 
                                    batch_size = batch_size, 
                                    callbacks = [callback])
                ncf_trained = True
            else:
                print('Run BNCF and TTV first!')

        elif file_io == 'ENCF' or file_io == 'encf':
            if ncf_trained == True:
                predictions = model.predict([test.user_id, test.item_id])
                preds_std = np.std(predictions)

                predictions_list = []
                for pred_rating in predictions:
                    predictions_list.append(pred_rating[0])

                ratings_preds_array = np.array(predictions_list).astype('float64')
                ratings_actual_array = np.array(test.rating)

                test_mse = sklearn.metrics.mean_squared_error(ratings_actual_array, ratings_preds_array)

                print(f'STD of predictions on test split: {preds_std}')
                print(f'MSE of predictions on test split: {test_mse}')
            
            else:
                print('Run TNCF first!')

        elif file_io == 'SNCF' or file_io == 'sncf':
            model.save('NCF_model')

        elif file_io == 'RNCF' or file_io == 'rncf':
            model = load_model('NCF_model')
            ncf_trained = True
        
        elif file_io == "RECS" or file_io == "recs":
            print()

            algo = input("Enter UU-CF, II-CF, MF-SGD, MF-ALS, NCF, TFIDF, Hybrid: ")
            userID = input(
                        "Enter username (for critics) or return to quit: "
                    )
            n_recs = int(input('Enter number of recommendations: '))

            if (len(prefs) > 0 and len(prefs) <= 10) or  (df.shape[0] > 0 and df.shape[0] <= 10):  # critics
                
                if algo == 'UU-CF' or algo == 'uu-cf':
                    sim_input = str(input("Enter similarity calculation - distance (d) or pearson (p): ")).lower()
                    threshold = float(input("Similarity threshold: "))
                    algo = getRecommendedUsers
                    matrix = usersim

                    if sim_input == 'd':
                        sim = sim_distance
                    else:
                        sim = sim_pearson

                    recs = algo(prefs, matrix, userID, threshold)

                elif algo == 'II-CF' or algo == 'ii-cf':
                    sim_input = str(input("Enter similarity calculation - distance (d) or pearson (p): ")).lower()
                    threshold = float(input("Similarity threshold: "))
                    algo = getRecommendedItems
                    matrix = itemsim

                    if sim_input == 'd':
                        sim = sim_distance
                    else:
                        sim = sim_pearson

                    recs = algo(prefs, matrix, userID, threshold)

                elif algo == 'MF-SGD' or algo == 'mf-sgd':
                    print('fix')
                
                elif algo == 'MF-ALS' or algo == 'mf-als':
                    print('fix')

                elif algo == 'NCF' or algo == 'ncf':
                    if ncf_trained == True:
                        
                        single_user_all_items_pairs_list = []
                        for i in range(n_items):
                            single_user_all_items_pairs_list.append([userID, i])

                        users_in_single_user_all_items_pairs = np.array(single_user_all_items_pairs_list)[:,0].reshape(-1,1)
                        items_in_single_user_all_items_pairs = np.array(single_user_all_items_pairs_list)[:,1].reshape(-1,1)
                        all_predictions = model.predict([users_in_single_user_all_items_pairs, items_in_single_user_all_items_pairs])
                        
                        recs = [(float(all_predictions[i]), movies[str(items_in_single_user_all_items_pairs[i])]) for i in range(len(items_in_single_user_all_items_pairs))]
                        print(f"recs for {userID}: {str(recs)}")
                
                elif algo == "TFIDF" or algo == "tfidf":
                    if tfidf_ran:
                        
                        if userID != "":
                            # Go run the TFIDF algo
                            threshold = float(input("Similarity threshold: "))
                            recs = get_TFIDF_recommendations(
                                prefs, cosim_matrix, userID, n_recs, movies, threshold
                            )
                            print(f"recs for {userID}: {str(recs)}")
                    else:
                        print("Run the TFIDF command first to set up TFIDF data")
                
                elif algo == "Hybrid" or algo == "hybrid":
                    if hybrid_ran & tfidf_ran & sim_ran:
                        userID = input(
                            "Enter username (for critics) or return to quit: "
                        )
                        if userID != "":
                            # Go run the hybrid algo
                            print("Go run the hybrid algo for %s" % userID)
                            n = int(input("Enter number of recommendations: "))
                            recs = get_hybrid_recommendations(
                                prefs,
                                updated_cosim_matrix,
                                userID,
                                n,
                                movies,
                                threshold,
                            )
                            print(f"recs for {userID}: {str(recs)}")
                    else:
                        print(
                            "Run the SIM, TFIDF, and HYB commands first to set up hybrid data"
                        )
                
                else:
                    print("Algorithm %s is invalid, try again!" % algo)

            elif len(prefs) > 10 or df.shape[0] > 10:

                if algo == 'UU-CF' or algo == 'uu-cf':
                    sim_input = str(input("Enter similarity calculation - distance (d) or pearson (p): ")).lower()
                    threshold = float(input("Similarity threshold: "))
                    algo = getRecommendedUsers
                    matrix = usersim

                    if sim_input == 'd':
                        sim = sim_distance
                    else:
                        sim = sim_pearson

                    recs = algo(prefs, matrix, userID, threshold)

                elif algo == 'II-CF' or algo == 'ii-cf':
                    sim_input = str(input("Enter similarity calculation - distance (d) or pearson (p): ")).lower()
                    threshold = float(input("Similarity threshold: "))
                    algo = getRecommendedItems
                    matrix = itemsim

                    if sim_input == 'd':
                        sim = sim_distance
                    else:
                        sim = sim_pearson

                    recs = algo(prefs, matrix, userID, threshold)

                elif algo == 'MF-SGD' or algo == 'mf-sgd':
                    print('fix')
                
                elif algo == 'MF-ALS' or algo == 'mf-als':
                    print('fix')

                elif algo == 'NCF' or algo == 'ncf':
                    if ncf_trained == True:
                        print('INSIDE')
                        single_user_all_items_pairs_list = []
                        for i in range(n_items):
                            single_user_all_items_pairs_list.append([int(userID), int(i)])
                        print('2')
                        users_in_single_user_all_items_pairs = np.array(single_user_all_items_pairs_list)[:,0].reshape(-1,1)
                        items_in_single_user_all_items_pairs = np.array(single_user_all_items_pairs_list)[:,1].reshape(-1,1)
                        print("users_in_single_user_all_items_pairs")
                        print(users_in_single_user_all_items_pairs)
                        print("items_in_single_user_all_items_pairs")
                        print(items_in_single_user_all_items_pairs)
                        print('3')
                        all_predictions = model.predict([users_in_single_user_all_items_pairs, items_in_single_user_all_items_pairs])
                        print('4')
                        recs = [(float(all_predictions[i]), movies[str(items_in_single_user_all_items_pairs[i][0] + 1)]) for i in range(len(items_in_single_user_all_items_pairs))]
                        recs.sort(reverse = True)

                        print(f"recs for {userID}: {str(recs[:n_recs])}")

                elif algo == "TFIDF" or algo == "tfidf":
                    if tfidf_ran:
                        if userID != "":
                            threshold = float(input("Similarity threshold: "))
                            # Go run the TFIDF algo
                            recs = get_TFIDF_recommendations(
                                prefs, cosim_matrix, userID, n_recs, movies, threshold
                            )
                            print(f"recs for {userID}: {str(recs)}")
                    else:
                        print("Run the TFIDF command first to set up TFIDF data")

                elif algo == "Hybrid" or algo == "hybrid":
                    if hybrid_ran & tfidf_ran & sim_ran:
                        userID = input("Enter userid (for ml-100k) or return to quit: ")
                        if userID != "":
                            # Go run the hybrid algo
                            threshold = float(input("Similarity threshold: "))
                            n = int(input("Enter number of recommendations: "))
                            recs = get_hybrid_recommendations(
                                prefs,
                                updated_cosim_matrix,
                                userID,
                                n,
                                movies,
                                threshold,
                            )
                            print(f"recs for {userID}: {str(recs)}")
                    else:
                        print(
                            "Run the SIM, TFIDF, and HYB commands first to set up hybrid data"
                        )

                else:
                    print("Algorithm %s is invalid, try again!" % algo)

            else:
                print("Empty dictionary, read in some data!")
                print()

        elif file_io == 'TFIDF-GRID' or file_io == 'tfidf-grid':
        
            tfidf_thresholds = [0, 0.3061939222, 0.5764484463073936, 0.8467029704]
            
            for tfidf_threshold in tfidf_thresholds:
                sim_sig_weighting = 0 # TFIDF doesn't use a sim sig weighting
                n_neighbors = 0 # TFIDF doesn't use n_neighbors
                algo = new_get_TFIDF_recommendations
                # Make cosim matrix
                R = to_array(prefs)
                feature_str = to_string(features)
                feature_docs = to_docs(feature_str, genres)
                cosim_matrix = cosine_sim(feature_docs)
                sim = 'TFIDF'
                mov = movies

                error_total, error_list = loo_cv_sim(
                    prefs,
                    sim,
                    algo,
                    cosim_matrix,
                    dataset_name,
                    tfidf_threshold,
                    sim_sig_weighting,
                    n_neighbors,
                    mov,
                )

        elif file_io == 'HYB-GRID' or file_io == 'hyb-grid':

            n_neighbors = 100
            sim_sig_weighting = 100 # best is 100 from midterm
            tfidf_thresholds = [0, 0.3061939222, 0.5764484463073936, 0.8467029704]
            weighting_factors = [.25, .5, .75, 1]
            sim_methods = [sim_distance, sim_pearson]

            for tfidf_threshold in tfidf_thresholds:
                for weighting_factor in weighting_factors:
                    for sim_method in sim_methods:
                    
                        # Make cosim matrix
                        R = to_array(prefs)
                        feature_str = to_string(features)
                        feature_docs = to_docs(feature_str, genres)
                        cosim_matrix = cosine_sim(feature_docs)

                        # Get II sim matrix
                        itemsim = calculateSimilarItems(
                            prefs,
                            neighbors=n_neighbors,
                            similarity=sim_method,
                            sim_sig_weighting=sim_sig_weighting,
                        )

                        algo = new_get_hybrid_recommendations
                        mov = movies

                        itemsim_mat = itemsim_to_np_matrix(itemsim, movies)
                        updated_cosim_matrix = hybrid_update_sim_matrix(
                            cosim_matrix, itemsim_mat, weighting_factor
                        )

                        error_total, error_list = loo_cv_sim(
                            prefs,
                            sim_method,
                            algo,
                            updated_cosim_matrix,
                            dataset_name,
                            tfidf_threshold,
                            sim_sig_weighting,
                            n_neighbors,
                            movies,
                            weighting_factor
                        )

        else:
            done_input = str(input('Are you finished? '))
            if done_input == 'Yes' or done_input == 'yes':
                done = True
            else:
                print('Try another command!')

    print("Goodbye!")


if __name__ == "__main__":
    main()
