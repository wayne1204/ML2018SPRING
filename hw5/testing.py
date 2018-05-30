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
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config=config))

parser = argparse.ArgumentParser(description='=== hw5 text sentiment analysis ===')
# training parameter
parser.add_argument('--strip')
parser.add_argument('--embed_dim', default=100, type = int)
parser.add_argument('--max_len', default=40, type = int)

# saving parameter
parser.add_argument('--binary')
parser.add_argument('--path')
args = parser.parse_args()

def test_data(fileName, strip):
    print('=== parsing file from %s ===' % fileName)
    X = []
    text = open(fileName, 'r')
    for line in text:
        sentence = line.strip().split(',', 1)[1]
        if strip == True:
            sentence = strip_punctuation(sentence)
        X.append(sentence.split())

    X = X[1:]
    print(X[:4])
    return X


# Keras Version
def word2seq(data,  max_len):
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pk.load(handle)
    sequence = tokenizer.texts_to_sequences(data)
    data = np.array(pad_sequences(sequence, maxlen=max_len, padding='post'))
    print("loading %i data with length %i" % (data.shape[0], data.shape[1]))
    return data

# genSim Version
def word2vec(data, path, word_dim, max_len):
    print('=== load genSim model ===')
    embed = gensim.models.Word2Vec.load(path)

    embedding_data = []
    for i in range(len(data)):
        row = []
        for j in range(max_len):
            if j < len(data[i]) and data[i][j] in embed.wv.vocab:
                row.append( embed[data[i][j]] )
            else:
                row.append( np.zeros(word_dim, ) )
        embedding_data.append(row)
    embedding_data = np.array(embedding_data)
    print(embedding_data.shape)
    return embedding_data

def prediction(data, path):
    print(data[0])
    print(data[0][0])
    print('=== predict from RNN model ===')
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
    EMBEDDING = args.embed_dim
    MAXLENGTH = args.max_len
    BINARY = args.binary
    MODEL = args.path
    if args.strip == 'True':
        STRIP = True
    else:
        STRIP = False

    X = test_data('data/testing_data.txt', STRIP)
    # X = word2seq(X, 40)
    X = word2vec(X, BINARY, EMBEDDING, MAXLENGTH)
    res = prediction(X, MODEL)
    outputAns(res, 'predict.csv')


if __name__ ==  "__main__":
    main()
    
