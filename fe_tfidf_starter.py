"""
Create content-based recommenders: Feature Encoding, TF-IDF/CosineSim
       using item/genre feature data
       

Programmer name: Brad Shook, Daniel Cowan, Drew Dibble


Collaborator/Author: Carlos Seminario

sources: 
https://www.freecodecamp.org/news/how-to-process-textual-data-using-tf-idf-in-python-cd2bbc0a94a3/
http://blog.christianperone.com/2013/09/machine-learning-cosine-similarity-for-vector-space-models-part-iii/
https://kavita-ganesan.com/tfidftransformer-tfidfvectorizer-usage-differences/#.XoT9p257k1L

references:
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

"""

from turtle import color
import numpy as np
import pandas as pd
import math
import os
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SIG_THRESHOLD = (
    0  # accept all positive similarities > 0 for TF-IDF/ConsineSim Recommender
)
# others: TBD ...


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


def get_TFIDF_recommendations(prefs, cosim_matrix, user, n, movies):
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
            if (sim >= SIG_THRESHOLD) and not (similar_item in prefs[user]):

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


def main():

    # Load critics dict from file
    path = os.getcwd()  # this gets the current working directory
    # you can customize path for your own computer here
    print("\npath: %s" % path)  # debug
    print()
    prefs = {}
    done = False

    while not done:
        print()
        file_io = input(
            "R(ead) critics data from file?, \n"
            "RML(ead) ml100K data from file?, \n"
            "TFIDF(and cosine sim Setup)?, \n"
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
            print('Reading "%s" dictionary from file' % datafile)
            prefs = from_file_to_dict(path, file_dir + datafile, file_dir + itemfile)
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

        elif file_io == "RML" or file_io == "rml":
            print()
            file_dir = "data/ml-100k/"  # path from current directory
            datafile = "u.data"  # ratngs file
            itemfile = "u.item"  # movie titles file
            genrefile = "u.genre"  # movie genre file
            print('Reading "%s" dictionary from file' % datafile)
            prefs = from_file_to_dict(path, file_dir + datafile, file_dir + itemfile)
            movies, genres, features = from_file_to_2D(
                path, file_dir + genrefile, file_dir + itemfile
            )

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

        elif file_io == "TFIDF" or file_io == "tfidf":
            print()
            # determine the U-I matrix to use ..
            if len(prefs) > 0 and len(prefs) <= 10:  # critics
                # convert prefs dictionary into 2D list
                R = to_array(prefs)
                feature_str = to_string(features)
                feature_docs = to_docs(feature_str, genres)

                """
                # e.g., critics data (CES)
                R = np.array([
                [2.5, 3.5, 3.0, 3.5, 2.5, 3.0],
                [3.0, 3.5, 1.5, 5.0, 3.5, 3.0],
                [2.5, 3.0, 0.0, 3.5, 0.0, 4.0],
                [0.0, 3.5, 3.0, 4.0, 2.5, 4.5],
                [3.0, 4.0, 2.0, 3.0, 2.0, 3.0],
                [3.0, 4.0, 0.0, 5.0, 3.5, 3.0],
                [0.0, 4.5, 0.0, 4.0, 1.0, 0.0],
                ])            
                """
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

                tfidf_ran = True

                """
                <class 'numpy.ndarray'> 
                
                [[1.         0.         0.35053494 0.         0.         0.61834884]
                [0.         1.         0.19989455 0.17522576 0.25156892 0.        ]
                [0.35053494 0.19989455 1.         0.         0.79459157 0.        ]
                [0.         0.17522576 0.         1.         0.         0.        ]
                [0.         0.25156892 0.79459157 0.         1.         0.        ]
                [0.61834884 0.         0.         0.         0.         1.        ]]
                """

                # plt.hist(cosim_matrix)
                # plt.show()
                # print and plot histogram of similarites

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

                tfidf_ran = True

                """
                <class 'numpy.ndarray'> 1682
                
                [[1.         0.         0.         ... 0.         0.34941857 0.        ]
                 [0.         1.         0.53676706 ... 0.         0.         0.        ]
                 [0.         0.53676706 1.         ... 0.         0.         0.        ]
                 [0.18860189 0.38145435 0.         ... 0.24094937 0.5397592  0.45125862]
                 [0.         0.30700538 0.57195272 ... 0.19392295 0.         0.36318585]
                 [0.         0.         0.         ... 0.53394963 0.         1.        ]]
                """

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
                # plt.vlines(x = 0.307, ymin = 0, ymax = 120000, color = 'r')
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

            else:
                print("Empty dictionary, read in some data!")
                print()

        elif file_io == "RECS" or file_io == "recs":
            print()
            # determine the U-I matrix to use ..

            if len(prefs) > 0 and len(prefs) <= 10:  # critics
                print("critics")
                algo = input("Enter TFIDF: ")
                if algo == "TFIDF" or algo == "tfidf":
                    if tfidf_ran:
                        userID = input(
                            "Enter username (for critics) or return to quit: "
                        )
                        if userID != "":
                            # Go run the TFIDF algo
                            print("Go run the TFIDF algo for %s" % userID)
                            n = int(input("Enter number of recommendations: "))
                            recs = get_TFIDF_recommendations(
                                prefs, cosim_matrix, userID, n, movies
                            )
                            print(f"recs for {userID}: {str(recs)}")
                    else:
                        print("Run the TFIDF command first to set up TFIDF data")
                else:
                    print("Algorithm %s is invalid, try again!" % algo)

            elif len(prefs) > 10:
                print("ml-100k")
                algo = input("Enter TFIDF: ")
                if algo == "TFIDF" or algo == "tfidf":
                    if tfidf_ran:
                        userID = input("Enter userid (for ml-100k) or return to quit: ")
                        if userID != "":
                            # Go run the TFIDF algo
                            print("Go run the TFIDF algo for %s" % userID)
                            n = int(input("Enter number of recommendations: "))
                            recs = get_TFIDF_recommendations(
                                prefs, cosim_matrix, userID, n, movies
                            )
                            print(f"recs for {userID}: {str(recs)}")
                    else:
                        print("Run the TFIDF command first to set up TFIDF data")

                else:
                    print("Algorithm %s is invalid, try again!" % algo)

            else:
                print("Empty dictionary, read in some data!")
                print()

        else:
            done = True

    print("Goodbye!")


if __name__ == "__main__":
    main()


"""

Sample output ..

RECS (for TFIDF)

ml-100k

Enter userid (for ml-100k) or return to quit: 340
rec for 340 =  [
(5.000000000000001, 'Wallace & Gromit: The Best of Aardman Animation (1996)'), 
(5.000000000000001, 'Faust (1994)'), 
(5.0, 'Woman in Question, The (1950)'), 
(5.0, 'Thin Man, The (1934)'), 
(5.0, 'Maltese Falcon, The (1941)'), 
(5.0, 'Lost Highway (1997)'), 
(5.0, 'Daytrippers, The (1996)'), 
(5.0, 'Big Sleep, The (1946)'), 
(4.823001861184155, 'Sword in the Stone, The (1963)'), 
(4.823001861184155, 'Swan Princess, The (1994)')]

"""