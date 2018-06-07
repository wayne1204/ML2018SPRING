import csv
import numpy as np
import gensim
import argparse
import _pickle as pk
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import strip_punctuation
from keras.models import Model, load_model

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

parser = argparse.ArgumentParser(description='=== hw5 text sentiment analysis ===')
# saving parameter
parser.add_argument('--fraction', default=0.3, type = float)
parser.add_argument('--load_path', default='model/model')
parser.add_argument('--test_data', default='data/testing_data.txt')
parser.add_argument('--ans', default='predict.csv')

# training parameter
parser.add_argument('--embed_dim', default=100, type = int)
parser.add_argument('--max_len', default=40, type = int)
args = parser.parse_args()

def test_data(fileName):
    print('=== parsing file from %s ===' % fileName)
    X = []
    text = open(fileName, 'r')
    for line in text:
        sentence = line.strip().split(',', 1)[1]
        X.append(sentence.split())

    X = X[1:]
    print(X[:4])
    return X

# genSim Version
def word2vec(data, path, word_dim, max_len):
    print('=== load genSim model ===')
    # embed = gensim.models.Word2Vec.load(path)
    embed = pk.load(open(path, 'rb')) 
    embedding_data = []
    for i in range(len(data)):
        print('padding row # {}/ {}\r'.format(i, len(data)), end='')
        row = []
        # data[i] = data[i][:max_len]
        for j in range(max_len):
            if j < len(data[i]) and data[i][j] in embed:
                row.append( embed[data[i][j]] )
            else:
                row.append( np.zeros(word_dim, ) )
        embedding_data.append(row)
    embedding_data = np.array(embedding_data)
    print(embedding_data.shape)
    return embedding_data

def prediction(data, path):
    # print(data[0])
    # print(data[0][0])
    print('=== predict from RNN model ===')
    model = load_model(path)
    result = model.predict(data, verbose = 1, batch_size=1024)
    return result

def outputAns(result, file):
    print('=== output answer file ===')
    result = np.argmax(result, axis=1)
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
    MAXLENGTH = args.max_len
    
    X = test_data(args.test_data)
    X_2 = word2vec(X, 'model/embed_3', 100, MAXLENGTH)
    res = prediction(X_2, 'model/RNN_model_3.h5')
    X_1 = word2vec(X, 'model/embed_4', 100, MAXLENGTH)
    res += prediction(X_1, 'model/RNN_model_4.h5')
    outputAns(res, args.ans)

if __name__ ==  "__main__":
    main()
    
