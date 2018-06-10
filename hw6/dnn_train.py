import numpy as np
import pandas as pd
import csv
import argparse

from util import *
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dot, add, Dropout, Concatenate
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.regularizers import l2

import keras.backend as K
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

def ArgumentParser():
    parser = argparse.ArgumentParser(description='=== hw6 movie recommendation ===')
    parser.add_argument('--users', default='data/users.csv')
    parser.add_argument('--movie', default='data/movies.csv')
    parser.add_argument('--fraction', default=0.4, type=float)
    parser.add_argument('--path', default='model/model_dnn.h5')
    
    # training parameter
    parser.add_argument('--dim', default=128, type=int)
    parser.add_argument('--drop', default=0.5, type=float)
    parser.add_argument('--val', default=0.05, type=float)
    parser.add_argument('--batch', default=512, type=int)
    return parser.parse_args()

def getModel(n_users, n_movies , args):
    user_in = Input(shape=(1, ))
    user = Embedding(n_users, args.dim)(user_in)
    user = Dropout(args.drop)(user)
    user = Flatten()(user)

    movie_in = Input(shape=(1, ))
    movie = Embedding(n_movies, args.dim)(movie_in)
    movie = Dropout(args.drop)(movie)
    movie = Flatten()(movie)

    # bias
    user_bias = Embedding(n_users, 1, embeddings_initializer='zeros')(user_in)
    user_bias = Flatten()(user_bias)
    movie_bias = Embedding(n_movies, 1, embeddings_initializer='zeros')(movie_in)
    movie_bias = Flatten()(movie_bias)
    
    # other feature
    gender_in = Input(shape=(1,))
    age_in = Input(shape=(7, ))
    occupy_in = Input(shape=(8, ))
    genre_in = Input(shape=(18, ))

    occupy_v = Dense(args.dim, activation='linear')(occupy_in)
    occupy_v = Dropout(args.drop)(occupy_v)
    genre_v = Dense(args.dim, activation='linear')(genre_in)
    genre_v = Dropout(args.drop)(genre_v)
    age_v = Dense(1, activation='linear')(age_in)

    # out
    dot1 = Dot(axes=1)([user, movie])
    dot2 = Dot(axes=1)([user, genre_v])
    dot3 = Dot(axes=1)([occupy_v, movie])
    dot4 = Dot(axes=1)([occupy_v, genre_v])
    dot5 = Dot(axes=1)([user, occupy_v])
    dot6 = Dot(axes=1)([movie, genre_v])
    output = Concatenate()([dot1, dot2, dot3, dot4, dot5, dot6,
             user_bias,movie_bias, gender_in, age_v])
    output = Dense(64, activation='selu')(output)
    output = BatchNormalization()(output)
    output = Dropout(0.5)(output)
    output = Dense(1, activation='linear')(output)
    model = Model(inputs=[user_in, gender_in, age_in, occupy_in, movie_in, genre_in], 
            outputs=output)

    def rmse(y_true, y_pred): return K.sqrt( K.mean((y_pred - y_true)**2) )
    adam = Adam(lr =1e-4)
    model.compile(loss='mse', optimizer=adam, metrics=[rmse])
    print(model.summary())
    return model

def main(args):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.fraction
    set_session(tf.Session(config=config))

    dm = DataManager()
    dm.parseUserData(args.users)
    dm.parseMovieData(args.movie)
    Users_Ids, Gender, Age, Occupy, Movie_Ids, Genre, Rating = dm.parseRating('data/train.csv')
    
    model = getModel(dm.n_users, dm.n_movie, args)
    checkpoint = ModelCheckpoint(args.path, monitor='val_rmse', verbose=1,
                                save_best_only=True, mode='min')
    early_stop = EarlyStopping(monitor='val_rmse',
                            min_delta=0,
                            patience=50,
                            verbose=1, mode='min')
    model.fit([Users_Ids, Gender, Age, Occupy, Movie_Ids, Genre], [Rating], 
                epochs=1000, batch_size = args.batch, validation_split=args.val, 
                callbacks=[checkpoint, early_stop])


if __name__ == '__main__':
    args = ArgumentParser()
    main(args)
