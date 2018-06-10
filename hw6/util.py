import numpy as np
import pandas as pd
import csv
import keras.backend as K

def shuffling(data):
    print(data[:10])
    print('Shuffle Data')
    np.random.seed(1)
    np.random.shuffle(data)
    print(data[:10])
    return data

def noramlizing(data):
    print('Normalizing rating')
    mean = np.mean(data)
    sigma = np.std(data)
    print('mean', mean)
    print('std', sigma)
    np.save('model/mean.npy', mean)
    np.save('model/sigma.npy', sigma)
    return (data - mean) / sigma

class DataManager():
    def __init__(self):
        self.n_users = 0
        self.n_movie = 0
        self.data = {}
        self.movieCategory = {}
        self.movieDict = {}

    def parseUserData(self, path):
        userData = []
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
                feature.append(int(line[2]))
                feature.append(int(line[3]))
                userData.append(feature)
        userData = sorted(userData, key = lambda x : int(x[0]))
        self.data['users'] = userData
        self.agetoVector()
        self.occupytoVector()
        self.n_users = len(userData) + 1
        print('=== finish parsing %i user data ===' % len(userData))

        
    def parseMovieData(self, path):
        movieData = []
        csvFile = open(path, encoding='latin-1')
        for i, row in enumerate(csvFile):
            if i != 0:
                movie = []
                row = row.strip().split('::')
                movie.append(int(row[0]))
                genres = row[2].split('|')
                for index, genre in enumerate(genres):
                    if genre not in self.movieCategory:
                        self.movieCategory[genre] = len(self.movieCategory)
                    genres[index] = self.movieCategory[genre]
                movie.append(genres)
                movieData.append(movie)
        print(self.movieCategory)
        self.data['movie'] = movieData
        self.genretoVector()
        self.n_movie = int(movieData[-1][0]) + 1
        print('=== finish parsing %i movie data ===' % len(movieData))

    def parseRating(self, path, lable = True):
        users, gender, age, occupy, movie, genre, rating = [], [], [], [], [], [], []

        data = pd.read_csv(path).values
        if lable:
            data = shuffling(data)
            rating = [data[i][3] for i in range(len(data))]
        for i in range(len(data)):
            uid = data[i][1]
            mid = data[i][2]
            users.append(uid)
            movie.append(mid)
            gender.append(self.data['users'][uid-1][1])
            age.append(self.data['users'][uid-1][2])
            occupy.append(self.data['users'][uid-1][3])
            genre.append(self.movieDict[mid])

        users = np.array(users)
        gender = np.array(gender)
        age = np.array(age)
        occupy = np.array(occupy)
        movie = np.array(movie)
        genre = np.array(genre)
        rating = np.array(rating)
        if lable:
            rating = noramlizing(rating)
        print('=== finish parsing %i training data ===' % len(users))
        return users, gender, age, occupy, movie, genre, rating

    def agetoVector(self):
        for i in range(len(self.data['users'])):
            Encode = np.zeros((7,))
            age = self.data['users'][i][2]
            if age <= 10:         Encode[0] = 1
            elif age <= 20:       Encode[1] = 1
            elif age <= 30:       Encode[2] = 1
            elif age <= 40:       Encode[3] = 1
            elif age <= 50:       Encode[4] = 1
            elif age <= 60:       Encode[5] = 1
            else: Encode[6] = 1
            self.data['users'][i][2] = Encode
             
    def occupytoVector(self):
        for i in range(len(self.data['users'])):
            Encode = np.zeros((8,))
            occupy = self.data['users'][i][3]
            if occupy == 0:       Encode[0] = 1
            elif occupy <= 2:     Encode[1] = 1
            elif occupy <= 5:     Encode[2] = 1
            elif occupy <= 9:     Encode[3] = 1
            elif occupy <= 14:    Encode[4] = 1
            elif occupy <= 19:    Encode[5] = 1
            elif occupy <= 25:    Encode[6] = 1
            else:    Encode[7] = 1
            self.data['users'][i][3] = Encode

    def genretoVector(self):
        length = len(self.data['movie'])
        for i in range(length):
            movieID = self.data['movie'][i][0]
            movieGenre = self.data['movie'][i][1]
            genreEncoding = np.zeros((18, ))
            for j in range(len(movieGenre)):
                genreEncoding[movieGenre[j]] = 1
            self.movieDict[movieID] = genreEncoding

