import numpy as np
import pandas as pd
import csv
import argparse

from util import *
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dot, add, Dropout
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2

import keras.backend as K
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

def ArgumentParser():
    parser = argparse.ArgumentParser(description='=== hw6 movie recommendation ===')
    parser.add_argument('--users', default='data/users.csv')
    parser.add_argument('--movie', default='data/movies.csv')
    parser.add_argument('--path', default='model/model.h5')
    
    # training parameter
    parser.add_argument('--dim', default=128, type=int)
    parser.add_argument('--drop', default=0.5, type=float)
    parser.add_argument('--val', default=0.05, type=float)
    parser.add_argument('--batch', default=10000, type=int)
    return parser.parse_args()

# def agetoVector(age):
#     ageEncode = np.zeros((7,))
#     age = int(age)
#     if age <= 10:         ageEncode[0] = 1
#     elif age <= 20:       ageEncode[1] = 1
#     elif age <= 30:       ageEncode[2] = 1
#     elif age <= 40:       ageEncode[3] = 1
#     elif age <= 50:       ageEncode[4] = 1
#     elif age <= 60:       ageEncode[5] = 1
#     else: ageEncode[6] = 1
#     return ageEncode

def genretoVector(movie):
    movieDict = {}
    for i in range(len(movie)):
        movieID = movie[i][0]
        movieGenre = movie[i][1]
        genreEncoding = np.zeros((18, ))
        for j in range(len(movieGenre)):
            genreEncoding[movieGenre[j]] = 1
        movieDict[movieID] = genreEncoding
    return movieDict

def getModel(n_users, n_movies , args):
    # , embeddings_regularizer = l2(0.00001
    u_1 = Input(shape=(1, ))
    u_2 = Embedding(n_users, args.dim)(u_1)
    u_2 = Dropout(args.drop)(u_2)
    u_3 = Flatten()(u_2)
    u_b = Embedding(n_users, 1, embeddings_initializer='zeros')(u_1)
    # u_b = Dropout(0.2)(u_b)
    u_b = Flatten()(u_b)

    m_1 = Input(shape=(1, ))
    m_2 = Embedding(n_movies, args.dim)(m_1)
    m_2 = Dropout(args.drop)(m_2)
    m_3 = Flatten()(m_2)
    m_b = Embedding(n_movies, 1, embeddings_initializer='zeros')(m_1)
    # m_b = Dropout(0.2)(m_b)
    m_b = Flatten()(m_b)

    dot = Dot(axes=1)([u_3, m_3])
    output = add([dot, u_b, m_b])
    model = Model(inputs=[u_1, m_1], outputs=output)

    def rmse(y_true, y_pred): return K.sqrt( K.mean((y_pred - y_true)**2) )
    model.compile(loss='mse', optimizer='adam', metrics=[rmse])
    print(model.summary())
    return model

def main(args):
    UserData = readUserData(args.users)
    MovieData = readMovieData(args.movie)
    n_Users = len(UserData) + 1
    n_Movies = int(MovieData[-1][0]) + 1
    # genre = genretoVector(Movie)

    Users_Ids, Movie_Ids, Rating = readRating('data/train.csv')
    model = getModel(n_Users, n_Movies, args)
    checkpoint = ModelCheckpoint(args.path, monitor='val_rmse', verbose=1,
                                save_best_only=True, mode='min')
    early_stop = EarlyStopping(monitor='val_rmse',
                               min_delta=0,
                               patience=50,
                               verbose=1, mode='min')
    model.fit([Users_Ids, Movie_Ids], [Rating], epochs=1000, batch_size = args.batch, validation_split=args.val, callbacks=[checkpoint, early_stop])


if __name__ == '__main__':
    args = ArgumentParser()
    main(args)
