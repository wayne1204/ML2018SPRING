import numpy as np
import csv
import sys
import math

import gensim
import argparse
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import strip_punctuation
from keras import regularizers
from keras.models import Model, Sequential, load_model
from keras.layers import BatchNormalization
from keras.layers import Input, LSTM, Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
import pickle as pk
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

parser = argparse.ArgumentParser(description='=== hw5 text sentiment analysis ===')
# I/O parameter
parser.add_argument('action', choices = ['train', 'semi'])
parser.add_argument('--fraction', default=0.5, type=float)
parser.add_argument('--train_data', default='data/training_label.txt')
parser.add_argument('--semi_data', default='data/training_nolabel.txt')
parser.add_argument('--binary', default='model/embed')
parser.add_argument('--load_path', default='None', type=str)
parser.add_argument('--save_path', default='model/RNN_model.h5')

# training parameter
parser.add_argument('--strip', help ='strip punctuations', type= str)
parser.add_argument('--embed_dim', default=100, type = int, help=' word embedding dimension')
parser.add_argument('--max_len', default=40, type = int, help=' max length of single sentence')
parser.add_argument('--min_count', default = 3, type = int, help=' min freq count in word2vec')
parser.add_argument('--wndw', default=5, type=int, help= 'window size in word2vec')
parser.add_argument('--iter', default=20, type=int, help=' iteration in word2vec')

args = parser.parse_args()

# each sentence is saved as list
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

# genSim version
def word2vec(data, embedName, word_dim=50, max_len=40, pretrain=False):
    if pretrain == True:
        embed = gensim.models.Word2Vec.load(embedName)
    else:
        embed = Word2Vec(data, word_dim, window = args.wndw, min_count = args.min_count, 
                iter = args.iter, workers = 8)
        embed.save(embedName)
    print('=== gensim parameter  ===')
    print(embed)

    embedding_data = []
    total_seq = 200000
    if args.action == 'semi':
        total_seq = 600000
    for i in range(total_seq):
        print('padding row # {}/ {}\r'.format(i, total_seq), end='')
        row = []
        for j in range(max_len):
            if j < len(data[i]) and data[i][j] in embed.wv.vocab:
                row.append( embed[data[i][j]] )
            else:
                row.append( np.zeros(word_dim, ) )
        embedding_data.append(row)
    embedding_data = np.array(embedding_data)
    np.save('embed_data', embedding_data)
    print('embedding data', embedding_data.shape)
    return embedding_data

def buildModel(embed_size, max_len):
    print('=== building model ===')
    model = Sequential()
    in_ = Input(shape=(max_len, embed_size))
    h = LSTM(256, dropout=0.2, recurrent_dropout=0.2,
                return_sequences=True, activation='tanh')(in_)
    h = LSTM(256, dropout=0.2, recurrent_dropout=0.2,
                return_sequences=True, activation='tanh')(h)
    h = LSTM(256, dropout=0.2, recurrent_dropout=0.2,
                return_sequences=False, activation='tanh')(h)
    # RNN
    # model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2,
    #             return_sequences=True, activation='tanh', input_shape=(max_len, embed_size)))
    # model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2, 
    #         return_sequences=True, activation='tanh'))
    # model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2,
    #         return_sequences=False, activation='tanh'))
    h = Dropout(0.3)(h)
    h = Dense(512, activation='relu')(h)
    h = BatchNormalization()(h)
    h = Dropout(0.4)(h)
    h = Dense(256, activation='relu')(h)
    h = BatchNormalization()(h)
    h = Dropout(0.4)(h)
    h = Dense(128, activation='relu')(h)
    h = BatchNormalization()(h)
    h = Dropout(0.4)(h)
    out_ = Dense(2, activation='softmax')(h)
    model = Model(inputs =in_, outputs=out_)

    # DNN
    # model.add(Dropout(0.3))
    # model.add(Dense(512, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.4))
    # model.add(Dense(256, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.4))
    # model.add(Dense(128, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.4))
    # model.add(Dense(2, activation='softmax'))

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
    model.fit([X_train], [Y_train], batch_size=1024, epochs=30,
              callbacks=[checkpoint, early_stop],
              validation_data=(X_valid, Y_valid)
              )

def self_training(model, X_semi):
    result = model.predict(X_semi, verbose = 1)
    X_new, Y_new = [], []
    for i in range(len(result)):
        if result[i][0] < 0.2:
            row = np.array([1, 0])
            X_new.append(X_semi[i])
            Y_new.append(row)
        elif result[i][0] > 0.8:
            row = np.array([0, 1])
            X_new.append(X_semi[i])
            Y_new.append(row)
    print('get new data of size {}'.format(len(X_new)))
    return X_new, Y_new

def main():
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.fraction
    set_session(tf.Session(config=config))
    EMBEDDING = args.embed_dim
    MAXLENGTH = args.max_len
    BINARY = args.binary
    PATH = args.save_path

    if args.strip == 'True':
        STRIP = True 
    else:
        STRIP = False

    X, Y = load_data(args.train_data, True, STRIP)
    X_semi, __ = load_data(args.semi_data, False, STRIP)
    print(len(X), len(X_semi))
    # check if pre-trained model
    if args.load_path != 'None':
        model = load_model(args.load_path)
        X_all = word2vec(X + X_semi, BINARY, EMBEDDING, MAXLENGTH, True)
    else:
        model = buildModel(EMBEDDING, MAXLENGTH)
        X_all = word2vec(X + X_semi, BINARY, EMBEDDING, MAXLENGTH, False)
    X_train = X_all[:200000]
    X_semi = X_all[200000:]
    # semi or not
    if args.action == 'train':
        supervised_training(model, PATH, X_train, Y)
    else:
        supervised_training(model, PATH, X_train, Y)
        for _ in range(10):
            X_new, Y_new = self_training(model, X_semi)
            X_new = np.concatenate((X_train, X_new), axis=0)
            Y_new = np.concatenate((Y, Y_new), axis=0)
            supervised_training(model, PATH, X_new, Y_new) 

if __name__ == "__main__":
    main()
