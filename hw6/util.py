import numpy as np
import pandas as pd
import csv
import keras.backend as K

def readUserData(path):
    data = []
    text = open(path, 'r')
    for i, line in enumerate(text):
        if i != 0:
            line = line.strip().split('::')
            feature = []
            feature.append(line[0])
            if line[1] == 'M':
                feature.append(1)
            else:
                feature.append(0)
            feature.append(line[2])
            data.append(feature)
    data = sorted(data, key= lambda x : int(x[0]))
    print(data[:5])
    print('=== finish parsing %i user data ===' % len(data))
    return data


def readMovieData(path):
    data = []
    genreDict = {}

    csvFile = open(path, encoding='latin-1')
    for i, row in enumerate(csvFile):
        if i != 0:
            movie = []
            row = row.strip().split('::')
            movie.append(row[0])
            genres = row[2].split('|')
            for index, genre in enumerate(genres):
                if genre not in genreDict:
                    genreDict[genre] = len(genreDict)
                genres[index] = genreDict[genre]
            movie.append(genres)
            data.append(movie)
    print('=== finish parsing %i movie data ===' % len(data))
    print(genreDict)
    return data 

def shuffling(data):
    print(data[:10])
    print('Shuffle Data')
    np.random.seed(1)
    data = np.random.shuffle(data)
    print(data[:10])
    return data

def noramlizing(data):
    print('Normalizing rating')
    mean = np.mean(data)
    sigma = np.std(data)
    np.save('model/mean.npy', mean)
    np.save('model/sigma.npy', sigma)
    return (data - mean) / sigma

def readRating(path, lable = True):
    users, movie, rating = [], [], []

    data = pd.read_csv(path)
    # data = shuffling(data)
    users = np.array(data['UserID'])
    movie = np.array(data['MovieID'])
    if lable == True:
        rating = np.array(data['Rating'])
        rating = noramlizing(rating)
    print('=== finish parsing %i training data ===' % len(users))
    return users, movie, rating
