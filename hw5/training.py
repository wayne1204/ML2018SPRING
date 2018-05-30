import numpy as np
import csv
import sys
import math

import gensim
import argparse
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import strip_punctuation
from keras import regularizers
from keras.models import Model, Sequential
from keras.layers import BatchNormalization
from keras.layers import Input, GRU, LSTM, Dense, Dropout, Bidirectional
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
import pickle as pk

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

parser = argparse.ArgumentParser(description='=== hw5 text sentiment analysis ===')
parser.add_argument('action', choices = ['train', 'semi'])

# training parameter
parser.add_argument('--strip', help ='strip punctuations', type= str)
parser.add_argument('--embed_dim', default=100, type = int, help=' word embedding dimension')
parser.add_argument('--max_len', default=40, type = int, help=' max length of single sentence')
parser.add_argument('--min_count', default = 3, type = int, help=' min freq count in word2vec')
parser.add_argument('--wndw', default=5, type=int, help= 'window size in word2vec')
parser.add_argument('--iter', default=16, type=int, help=' iteration in word2vec')

# saving parameter
parser.add_argument('--binary')
parser.add_argument('--save_dir')
args = parser.parse_args()

def load_data(fileName, label = True, strip = True):
    print('=== parsing data from %s ===' % fileName)
    X, Y= [], []
    text = open(fileName, 'r')
    for line in text:
        if label == True:
            sentence = line.strip().split(' +++$+++ ')
            Y.append(int(sentence[0]))
            if strip == True:
                sentence[1] = strip_punctuation(sentence[1])
            X.append(sentence[1].split())
        else:
            sentence = line.strip().split()
            X.append(sentence)
    print(X[:4])
    Y = np.array(to_categorical(Y, 2))
    return X, Y


def splitValidation(X, Y, percentage):
    print('=== spliting validation data ===')
    np.random.seed(0)
    randomize = np.arange(len(X))
    randomize = np.random.shuffle(randomize)
    v_size = int(math.floor(len(X) * percentage))
    X_train, Y_train = X[0:v_size], Y[0:v_size]
    X_valid, Y_valid = X[v_size:], Y[v_size:]
    return X_train, Y_train, X_valid, Y_valid

# Keras Version
def word2seq(voc_size, data,  max_len):
    tokenizer = Tokenizer(num_words=voc_size)
    tokenizer.fit_on_texts(data)
    sequence = tokenizer.texts_to_sequences(data)
    data = np.array(pad_sequences(sequence, maxlen=max_len, padding='post'))
    print("lexicon size = %i" % voc_size)
    print("loading %i data with length %i" % (data.shape[0], data.shape[1]))
    with open('tokenizer.pkl', 'wb') as handler:
        pk.dump(tokenizer, handler)
    return voc_size, data

# genSim version
def word2vec(data, embedName, word_dim=50, max_len=40):
    embed = Word2Vec(data, word_dim, window = args.wndw, min_count = args.min_count, 
        iter = args.iter, workers = 8)
    print('=== gensim parameter  ===')
    print(embed)
    embed.save(embedName)

    embedding_data = []
    total_seq = 200000
    if args.action == 'semi':
        total_seq = len(data)
    for i in range(total_seq):
        row = []
        for j in range(max_len):
            if j < len(data[i]) and data[i][j] in embed.wv.vocab:
                row.append( embed[data[i][j]] )
            else:
                row.append( np.zeros(word_dim, ) )
        embedding_data.append(row)
    embedding_data = np.array(embedding_data)
    print('embedding data', embedding_data.shape)
    return embedding_data

def buildModel(embed_size, max_len):
    model = Sequential()

    # RNN
    model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2,
                return_sequences=True, activation='tanh', input_shape=(max_len, embed_size)))
    model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2, 
            return_sequences=True, activation='tanh'))
    model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2,
            return_sequences=False, activation='tanh'))

    # DNN
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer='Adam', metrics=['accuracy'])
    print(model.summary())
    return model

def supervised_training(model, path, X, Y):
    X_train, Y_train, X_valid, Y_valid = splitValidation(X, Y, 0.9)
    checkpoint = ModelCheckpoint(path, monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='max')
    early_stop = EarlyStopping(monitor='val_acc',
                               min_delta=0,
                               patience=5,
                               verbose=1, mode='max')
    model.fit(X_train, Y_train, batch_size=256, epochs=50,
              callbacks=[checkpoint, early_stop],
              validation_data=(X_valid, Y_valid)
              )

# def self_training(model, X, Y):


def main():
    training = 'data/training_label.txt'
    semi = 'data/training_nolabel.txt'
    EMBEDDING = args.embed_dim
    MAXLENGTH = args.max_len
    BINARY = args.binary
    PATH = args.save_dir

    if args.strip == 'True':
        STRIP = True 
    else:
        STRIP = False

    X, Y = load_data(training, True, STRIP)
    X_semi, Y_semi = load_data(semi, False, STRIP)
    print(len(X), len(X_semi))
    X = X + X_semi
    X = word2vec(X, BINARY, EMBEDDING, MAXLENGTH)
    model = buildModel(EMBEDDING, MAXLENGTH)
    supervised_training(model, PATH, X, Y)

if __name__ == "__main__":
    main()
