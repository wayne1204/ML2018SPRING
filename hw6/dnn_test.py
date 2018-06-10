import numpy as np
import argparse
import csv
from util import *

from keras.models import load_model
import keras.backend as K
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

def ArgumentParser():
    parser = argparse.ArgumentParser(
        description='=== hw6 movie recommendation ===')
    parser.add_argument('--users', default='data/users.csv')
    parser.add_argument('--movie', default='data/movies.csv')
    parser.add_argument('--testdata', default='data/test.csv')
    parser.add_argument('--fraction', default=0.4, type=float)
    parser.add_argument('--path', default='model/model_dnn_1.h5')
    
    parser.add_argument('action', choices=['test', 'ensemble'], default='test')
    parser.add_argument('--ans', default='predict.csv')
    return parser.parse_args()

def rmse(y_true, y_pred): return K.sqrt( K.mean((y_pred - y_true)**2) )

def mf_predict(model_path, user, movie):
    print('=== load model from %s ===' % model_path)
    mean = np.load('model/mean.npy')
    sigma = np.load('model/sigma.npy')
    model = load_model(model_path, custom_objects={'rmse': rmse})
    result = model.predict([user, movie], verbose=1)
    result = result * sigma + mean
    print(np.max(result))
    print(np.min(result))
    return result

def prediction(model_path, user, gender, age, occupy, movie, genre):
    print('=== load model from %s ===' % model_path)
    mean = np.load('model/mean.npy')
    sigma = np.load('model/sigma.npy')
    model = load_model(model_path, custom_objects={'rmse': rmse})
    result = model.predict([user, gender, age, occupy, movie, genre], verbose=1)
    result = result * sigma + mean
    print(result[:5])
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

    users_IDs, Gender, Age, Occupy, movie_IDs, Genre, Rating  = dm.parseRating(args.testdata, False)   
    model_path = args.path
    ans = prediction(model_path, users_IDs, Gender, Age, Occupy, movie_IDs, Genre)
    if args.action == 'ensemble':
        ans += prediction('model/model_dnn_3.h5', users_IDs, Gender, Age, Occupy, movie_IDs, Genre)
        ans += prediction('model/model_dnn_4.h5', users_IDs, Gender, Age, Occupy, movie_IDs, Genre)
        ans += prediction('model/model_dnn_5.h5', users_IDs, Gender, Age, Occupy, movie_IDs, Genre)
        ans += prediction('model/model_dnn_6.h5', users_IDs, Gender, Age, Occupy, movie_IDs, Genre)
        ans += prediction('model/model_dnn_7.h5', users_IDs, Gender, Age, Occupy, movie_IDs, Genre)
        ans /= 6
    outputAns(ans, args.ans)


if __name__ == '__main__':
    args = ArgumentParser()
    main(args)