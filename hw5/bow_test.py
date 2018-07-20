import csv
import numpy as np
import pickle as pk
import gensim
import argparse
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import strip_punctuation
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


parser = argparse.ArgumentParser(description='=== hw5 text sentiment analysis ===')
# I/O parameter
parser.add_argument('--fraction', default=0.3, type=float)
parser.add_argument('--binary')
parser.add_argument('--load_path', default='model/model')
parser.add_argument('--test_data', default='data/testing_data.txt')
parser.add_argument('--ans', default='predict.csv')
args = parser.parse_args()

def test_data(fileName):
    print('=== parsing file from %s ===' % fileName)
    X = []
    text = open(fileName, 'r')
    for line in text:
        sentence = line.strip().split(',', 1)[1]
        X.append(sentence)
    X = X[1:]
    print(X[:4])
    return X


# Keras bag of word
def word2matrix(data, path):
    with open(path, 'rb') as handle:
        tokenizer = pk.load(handle)
    data = tokenizer.texts_to_matrix(data)
    print("loading %i data with length %i" % (data.shape[0], data.shape[1]))
    return data

def prediction(data, path):
    print('=== predict from BOW model ===')
    model = load_model(path)
    result = model.predict(data, verbose = 1)
    result = np.argmax(result, axis=1)
    return result

def outputAns(result, file):
    print('=== output answer file ===')
    ans = []
    for i in range(len(result)):
        ans.append([i])
        ans[i].append( int(result[i]) )

    text = open(file, "w+")
    s = csv.writer(text, delimiter=',', lineterminator='\n')
    s.writerow(["id", "label"])
    for i in range(len(ans)):
	    s.writerow(ans[i])

def main():
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.fraction
    set_session(tf.Session(config=config))

    X = test_data(args.test_data)
    X = word2matrix(X, args.binary)
    res = prediction(X, args.load_path)
    outputAns(res, args.ans)


if __name__ ==  "__main__":
    main()
    
