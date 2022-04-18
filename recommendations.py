"""
CSC381: Building a simple Recommender System
The final code package is a collaborative programming effort between the
CSC381 student(s) named below, the class instructor (Carlos Seminario), and
source code from Programming Collective Intelligence, Segaran 2007.
This code is for academic use/purposes only.
CSC381 Programmer/Researcher: Brad Shook, Daniel Cowan, Henry Waddill
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
    plt.hist(ratings_list, bins=4, range=(1, 5), color='#ac1a2f')
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
def sim_distance(prefs, person1, person2, sig_weight_cutoff):
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

    if len(si) < sig_weight_cutoff and sig_weight_cutoff != 1:
        return similarity * (len(si) / sig_weight_cutoff)

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
def sim_pearson(prefs, p1, p2, sig_weight_cutoff):
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

    if len(si) < sig_weight_cutoff and sig_weight_cutoff != 1:
        return r * (len(si) / sig_weight_cutoff)

    return r


def sim_tanimoto(prefs, p1, p2, sig_weight_cutoff):
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

    if len(si) < sig_weight_cutoff and sig_weight_cutoff != 1:
        return r * (len(si) / sig_weight_cutoff)

    return r


def sim_jaccard(prefs, p1, p2, sig_weight_cutoff):

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

    if len(common_items) < sig_weight_cutoff and sig_weight_cutoff != 1:
        return jaccard_index * (len(common_items) / sig_weight_cutoff)

    return jaccard_index


def sim_cosine(prefs, p1, p2, sig_weight_cutoff):
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

    if len(si) < sig_weight_cutoff and sig_weight_cutoff != 1:
        return r * (len(si) / sig_weight_cutoff)

    return r


def sim_spearman(prefs, p1, p2, sig_weight_cutoff):
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

    if len(si) < sig_weight_cutoff and sig_weight_cutoff != 1:
        return coef * (len(si) / sig_weight_cutoff)

    return coef


def sim_kendall_tau(prefs, p1, p2, sig_weight_cutoff):
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

    if len(si) < sig_weight_cutoff and sig_weight_cutoff != 1:
        return coef * (len(si) / sig_weight_cutoff)

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


def topMatches(prefs, person, similarity=sim_pearson, n=100, sig_weight_cutoff=1):
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
        (similarity(prefs, person, other, sig_weight_cutoff), other)
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
    prefs, neighbors=100, similarity=sim_pearson, sig_weight_cutoff=1
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
            sig_weight_cutoff=sig_weight_cutoff,
        )
        result[item] = scores
    return result


def calculateSimilarUsers(
    prefs, neighbors=100, similarity=sim_pearson, sig_weight_cutoff=1
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
            prefs, user, similarity, n=neighbors, sig_weight_cutoff=sig_weight_cutoff
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


def new_getRecommendedItems(prefs, itemMatch, user, threshold, cur_item):
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


def new_getRecommendedUsers(prefs, userMatch, user, threshold, cur_item):
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
    prefs, sim, algo, sim_matrix, dataset_name, threshold, weight, neighbors
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

            del prefs[user][item]

            prediction = algo(prefs, sim_matrix, user, threshold, item)
            # prediction = algo(prefs, sim_matrix, user, threshold)

            # prediction = [rec for rec in recs if item in rec]

            if prediction != []:
                curr_error = prediction[0][0] - removed_rating
                error_list.append(curr_error)

            prefs[user][item] = removed_rating

    secs = time.time() - start
    print(f"Number of users processed: {i}")
    print(f"==>> {secs} secs for {i} users, secs per user {secs/i}")

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
    print(
        f"MSE for {dataset_name}: {round(mse,5)}, len(SE list): {len(error_array)} using {sim}"
    )

    coverage = len(error_array) / sum([len(prefs[person].values()) for person in prefs])

    sim_str = str(sim).split()[1]
    algo_str = str(algo).split()[1]

    cur_res_dict = {
        "dataset_name": [dataset_name],
        "sim_method": [sim_str],
        "algo": [algo_str],
        "sim_threshold": [threshold],
        "neighbors": [neighbors],
        "sig_weight": [weight],
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


def main():
    """User interface for Python console"""

    # Load critics dict from file
    path = os.getcwd()  # this gets the current working directory
    # you can customize path for your own computer here
    print("\npath: %s" % path)  # debug
    done = False
    prefs = {}

    n_neighbors = int(input("Number of neighbors: "))
    threshold = float(input("Similarity threshold: "))
    weight = int(input("Significance weighting cutoff: "))

    while not done:
        print()
        # Start a simple dialog
        file_io = input(
            "R(ead) critics data from file?, \n"
            "RML(ead) Movie Lens data from file?, \n"
            "P(rint) the U-I matrix?, \n"
            "V(alidate) the dictionary?, \n"
            "S(tats) print?, \n"
            "D(istance) critics data?, \n"
            "PC(earson Correlation) critics data?, \n"
            "U(ser-based CF Recommendations)?, \n"
            "LCV(eave one out cross-validation)? \n"
            "Sim(ilarity matrix) calc for Item-based recommender? \n"
            "Simu(ser similarity matrix) calc for User-based recommender? \n"
            "I(tem-based CF Recommendations)?, \n"
            "LCVSIM(eave one out cross-validation)? \n"
            "REC(ommendations, Item or User CF)? ==> "
        )

        if file_io == "R" or file_io == "r":
            print()
            file_dir = "data/"
            datafile = "critics_ratings.data"
            itemfile = "critics_movies.item"
            dataset_name = "critics"
            print('Reading "%s" dictionary from file' % datafile)
            prefs = from_file_to_dict(path, file_dir + datafile, file_dir + itemfile)
            print(
                "Number of users: %d\nList of users:" % len(prefs), list(prefs.keys())
            )

        elif file_io == "RML" or file_io == "rml":
            print()
            file_dir = "data/ml-100k/"
            datafile = "u.data"
            itemfile = "u.item"
            dataset_name = "ml-100k"
            print('Reading "%s" dictionary from file' % datafile)
            prefs = from_file_to_dict(path, file_dir + datafile, file_dir + itemfile)
            print("Number of users: %d" % len(prefs))
            print(f"List of users [0:10]: {list(prefs.keys())[:10]}")

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
                try:
                    if sub_cmd == "RD" or sub_cmd == "rd":
                        # Load the dictionary back from the pickle file.
                        itemsim = pickle.load(
                            open(
                                f"sim_mat/save_itemsim_distance_{dataset_name}.p", "rb"
                            )
                        )
                        sim_method = "sim_distance"

                    elif sub_cmd == "WD" or sub_cmd == "wd":
                        # transpose the U-I matrix and calc item-item similarities matrix
                        itemsim = calculateSimilarItems(
                            prefs,
                            neighbors=n_neighbors,
                            similarity=sim_distance,
                            sig_weight_cutoff=weight,
                        )
                        # Dump/save dictionary to a pickle file
                        pickle.dump(
                            itemsim,
                            open(
                                f"sim_mat/save_itemsim_distance_{dataset_name}.p", "wb"
                            ),
                        )
                        sim_method = "sim_distance"

                    elif sub_cmd == "RP" or sub_cmd == "rp":
                        # Load the dictionary back from the pickle file.
                        itemsim = pickle.load(
                            open(f"sim_mat/save_itemsim_pearson_{dataset_name}.p", "rb")
                        )
                        sim_method = "sim_pearson"

                    elif sub_cmd == "WP" or sub_cmd == "wp":
                        # transpose the U-I matrix and calc item-item similarities matrix
                        itemsim = calculateSimilarItems(
                            prefs,
                            neighbors=n_neighbors,
                            similarity=sim_pearson,
                            sig_weight_cutoff=weight,
                        )
                        # Dump/save dictionary to a pickle file
                        pickle.dump(
                            itemsim,
                            open(
                                f"sim_mat/save_itemsim_pearson_{dataset_name}.p", "wb"
                            ),
                        )
                        sim_method = "sim_pearson"

                    elif sub_cmd == "RT" or sub_cmd == "rt":
                        # Load the dictionary back from the pickle file.
                        itemsim = pickle.load(
                            open(
                                f"sim_mat/save_itemsim_tanimoto_{dataset_name}.p", "rb"
                            )
                        )
                        sim_method = "sim_tanimoto"

                    elif sub_cmd == "WT" or sub_cmd == "wt":
                        # transpose the U-I matrix and calc item-item similarities matrix
                        itemsim = calculateSimilarItems(
                            prefs,
                            neighbors=n_neighbors,
                            similarity=sim_tanimoto,
                            sig_weight_cutoff=weight,
                        )
                        # Dump/save dictionary to a pickle file
                        pickle.dump(
                            itemsim,
                            open(
                                f"sim_mat/save_itemsim_tanimoto_{dataset_name}.p", "wb"
                            ),
                        )
                        sim_method = "sim_tanimoto"

                    elif sub_cmd == "RJ" or sub_cmd == "rj":
                        # Load the dictionary back from the pickle file.
                        itemsim = pickle.load(
                            open(f"sim_mat/save_itemsim_jaccard_{dataset_name}.p", "rb")
                        )
                        sim_method = "sim_jaccard"

                    elif sub_cmd == "WJ" or sub_cmd == "wj":
                        # transpose the U-I matrix and calc item-item similarities matrix
                        itemsim = calculateSimilarItems(
                            prefs,
                            neighbors=n_neighbors,
                            similarity=sim_jaccard,
                            sig_weight_cutoff=weight,
                        )
                        # Dump/save dictionary to a pickle file
                        pickle.dump(
                            itemsim,
                            open(
                                f"sim_mat/save_itemsim_jaccard_{dataset_name}.p", "wb"
                            ),
                        )
                        sim_method = "sim_jaccard"

                    elif sub_cmd == "RC" or sub_cmd == "rc":
                        # Load the dictionary back from the pickle file.
                        itemsim = pickle.load(
                            open(f"sim_mat/save_itemsim_cosine_{dataset_name}.p", "rb")
                        )
                        sim_method = "sim_cosine"

                    elif sub_cmd == "WC" or sub_cmd == "wc":
                        # transpose the U-I matrix and calc item-item similarities matrix
                        itemsim = calculateSimilarItems(
                            prefs,
                            neighbors=n_neighbors,
                            similarity=sim_cosine,
                            sig_weight_cutoff=weight,
                        )
                        # Dump/save dictionary to a pickle file
                        pickle.dump(
                            itemsim,
                            open(f"sim_mat/save_itemsim_cosine_{dataset_name}.p", "wb"),
                        )
                        sim_method = "sim_cosine"

                    elif sub_cmd == "RS" or sub_cmd == "rs":
                        # Load the dictionary back from the pickle file.
                        itemsim = pickle.load(
                            open(
                                f"sim_mat/save_itemsim_spearman_{dataset_name}.p", "rb"
                            )
                        )
                        sim_method = "sim_spearman"

                    elif sub_cmd == "WS" or sub_cmd == "ws":
                        # transpose the U-I matrix and calc item-item similarities matrix
                        itemsim = calculateSimilarItems(
                            prefs,
                            neighbors=n_neighbors,
                            similarity=sim_spearman,
                            sig_weight_cutoff=weight,
                        )
                        # Dump/save dictionary to a pickle file
                        pickle.dump(
                            itemsim,
                            open(
                                f"sim_mat/save_itemsim_spearman_{dataset_name}.p", "wb"
                            ),
                        )
                        sim_method = "sim_spearman"

                    elif sub_cmd == "RK" or sub_cmd == "rk":
                        # Load the dictionary back from the pickle file.
                        itemsim = pickle.load(
                            open(
                                f"sim_mat/save_itemsim_kendall_tau_{dataset_name}.p",
                                "rb",
                            )
                        )
                        sim_method = "sim_kendall_tau"

                    elif sub_cmd == "WK" or sub_cmd == "wk":
                        # transpose the U-I matrix and calc item-item similarities matrix
                        itemsim = calculateSimilarItems(
                            prefs,
                            neighbors=n_neighbors,
                            similarity=sim_kendall_tau,
                            sig_weight_cutoff=weight,
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
                            sig_weight_cutoff=weight,
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
                            sig_weight_cutoff=weight,
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
                            sig_weight_cutoff=weight,
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
                            sig_weight_cutoff=weight,
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
                            sig_weight_cutoff=weight,
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
                            sig_weight_cutoff=weight,
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
                            sig_weight_cutoff=weight,
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

        elif file_io == "I" or file_io == "i":
            print()
            if len(prefs) > 0 and len(itemsim) > 0:
                print("Example:")
                user_name = "Toby"

                print(
                    "Item-based CF recs for %s, %s: " % (user_name, sim_method),
                    getRecommendedItems(prefs, itemsim, user_name, threshold, weight),
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
                        "User-based (user) or item-based (item) recommendations? "
                    )

                    if input_algo.lower() == "user":
                        algo = new_getRecommendedUsers
                        sim_matrix = usersim
                    elif input_algo.lower() == "item":
                        algo = new_getRecommendedItems  # Item-based recommendation
                        sim_matrix = itemsim
                    else:
                        print(
                            'Invalid recommendation algo. Please say "user" or "item".'
                        )
                        return

                    if sim_method == "sim_pearson":
                        sim = sim_pearson
                        error_total, error_list = loo_cv_sim(
                            prefs,
                            sim,
                            algo,
                            sim_matrix,
                            dataset_name,
                            threshold,
                            weight,
                            n_neighbors,
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
                            weight,
                            n_neighbors,
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
                            weight,
                            n_neighbors,
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
                            weight,
                            n_neighbors,
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
                            weight,
                            n_neighbors,
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
                            weight,
                            n_neighbors,
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
                            weight,
                            n_neighbors,
                        )

                        print()

                    else:
                        print(
                            "Run Sim(ilarity matrix) command to create/load Sim matrix!"
                        )
                    if dataset_name == "critics":
                        print(error_list)

                    # for anova testing purposes
                    sim_str = str(sim).split()[1]
                    algo_str = str(algo).split()[1]
                    pickle.dump(
                        error_list,
                        open(
                            f"errors_brad/sq_errors_{dataset_name}_{str(threshold).replace('.',',')}_{weight}_{sim_str}_{algo_str}_{n_neighbors}.p",
                            "wb",
                        ),
                    )

                else:
                    print("Empty dictionary, run R(ead) OR Empty Sim Matrix, run Sim!")
            except Exception as ex:
                print(f"{ex}: Empty Sim Matrix, run Sim!")
                tb = traceback.format_exc()
                print(str(tb))

        elif file_io == "REC" or file_io == "rec":
            algo_input = str(input("Enter algo - user (u) or item (i): ")).lower()
            sim_input = str(
                input("Enter similarity calculation - distance (d) or pearson (p): ")
            ).lower()
            user = str(input("Enter User name/identifier: "))
            num_recs = int(input("Enter number of recommendations: "))
            try:
                if algo_input == "u":
                    algo = getRecommendedUsers
                    matrix = usersim
                else:
                    algo = getRecommendedItems
                    matrix = itemsim
            except Exception as ex:
                print(f"Error: {ex}. Try running sim or simu command first.")

            if sim_input == "d":
                sim = sim_distance
            else:
                sim = sim_pearson

            recs = algo(prefs, matrix, user, 0, 1)

            print(f"CF recs using {algo} for {user}: {recs[:num_recs]}")

        else:
            done = True

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
