import numpy as np
import pandas as pd
import csv
import argparse

from util import *
from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten, Dot, add, Dropout
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2

import keras.backend as K
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

def ArgumentParser():
    parser = argparse.ArgumentParser(description='=== hw6 movie recommendation ===')
    parser.add_argument('action', choices=['train', 'test'])
    parser.add_argument('--users', default='data/users.csv')
    parser.add_argument('--movie', default='data/movies.csv')
    parser.add_argument('--testdata', default='data/test.csv')
    parser.add_argument('--fraction', default=0.4, type=float)
    parser.add_argument('--path', default='model/mf_model_128.h5')
    parser.add_argument('--ans', default='predict.csv')
    
    # training parameter
    parser.add_argument('--dim', default=128, type=int)
    parser.add_argument('--drop', default=0.5, type=float)
    parser.add_argument('--val', default=0.05, type=float)
    parser.add_argument('--batch', default=10000, type=int)
    return parser.parse_args()

def rmse(y_true, y_pred): return K.sqrt( K.mean((y_pred - y_true)**2) )

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

    # dot
    dot = Dot(axes=1)([user, movie])
    output = add([dot, user_bias, movie_bias])
    model = Model(inputs=[user_in, movie_in], outputs=output)

    model.compile(loss='mse', optimizer='adam', metrics=[rmse])
    print(model.summary())
    return model

def prediction(model_path, user, movie):
    print('=== load model from %s ===' % model_path)
    mean = np.load('model/mean.npy')
    sigma = np.load('model/sigma.npy')
    print(mean, sigma)
    model = load_model(model_path, custom_objects={'rmse': rmse})
    result = model.predict([user, movie], verbose=1)
    result = result * sigma + mean
    print(np.max(result))
    print(np.min(result))
    return result

def outputAns(result, fileName):
    print('=== output answer file ===')
    result = np.clip(result, 1, 5)
    ans = []
    for i in range(len(result)):
        ans.append([i+1])
        ans[i].append(result[i][0])

    text = open(fileName, "w+")
    s = csv.writer(text, delimiter=',', lineterminator='\n')
    s.writerow(["TestDataID", "Rating"])
    for i in range(len(ans)):
	    s.writerow(ans[i])

def main(args):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.fraction
    set_session(tf.Session(config=config))

    dm = DataManager()
    dm.parseUserData(args.users)
    dm.parseMovieData(args.movie)
    # training
    if args.action == 'train':
        Users_Ids, Gender, Age, Occupy, Movie_Ids, Genre, Rating = dm.parseRating('data/train.csv')
        model = getModel(dm.n_users, dm.n_movie, args)
        checkpoint = ModelCheckpoint(args.path, monitor='val_rmse', verbose=1,
                                    save_best_only=True, mode='min')
        early_stop = EarlyStopping(monitor='val_rmse',
                                min_delta=0,
                                patience=50,
                                verbose=1, mode='min')
        model.fit([Users_Ids, Movie_Ids], [Rating], epochs=1000, batch_size = args.batch, validation_split=args.val, callbacks=[checkpoint, early_stop])
    # testing
    else:
        users_IDs, Gender, Age, Occupy, movie_IDs, Genre, _  = dm.parseRating(args.testdata, False)   
        ans = prediction(args.path, users_IDs, movie_IDs)
        outputAns(ans, args.ans)

if __name__ == '__main__':
    args = ArgumentParser()
    main(args)
