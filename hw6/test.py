# import pandas as pd
import numpy as np
import argparse
import csv
from util import readRating

from keras.models import load_model
import keras.backend as K
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

parser = argparse.ArgumentParser(
    description='=== hw6 movie recommendation ===')
parser.add_argument('--path', default='model/model_1.h5')
parser.add_argument('--ans', default='predict.csv')
args = parser.parse_args()

def rmse(y_true, y_pred): return K.sqrt( K.mean((y_pred - y_true)**2) )

def prediction(model_path, user, movie):
    print('=== load model from %s ===' % model_path)
    mean = np.load('model/mean.npy')
    sigma = np.load('model/sigma.npy')
    model = load_model(model_path, custom_objects={'rmse': rmse})
    result = model.predict([user, movie], verbose=1)
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

def main():
    users_IDs, movie_IDs, _r = readRating('data/test.csv', False)
    model_path = args.path
    ans = prediction(model_path, users_IDs, movie_IDs)
    ans += prediction('model/model_2.h5', users_IDs, movie_IDs)
    ans += prediction('model/model_3.h5', users_IDs, movie_IDs)
    ans += prediction('model/model_4.h5', users_IDs, movie_IDs)
    ans += prediction('model/model_5.h5', users_IDs, movie_IDs)
    ans /= 5
    outputAns(ans, args.ans)


if __name__ == '__main__':
    main()
