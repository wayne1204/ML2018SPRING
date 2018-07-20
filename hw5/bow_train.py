import numpy as np
import csv
import sys
import math
import argparse
from keras import regularizers
from keras.models import Model, Sequential
from keras.layers import BatchNormalization
from keras.layers import Input, Dense, Dropout, Bidirectional
import matplotlib.pyplot as plt
from keras.layers.embeddings import Embedding
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
# I/O parameter
# parser.add_argument('action', choices = ['train', 'semi'])
parser.add_argument('--train_data', default='data/training_label.txt')
parser.add_argument('--semi_data', default='data/training_nolabel.txt')
parser.add_argument('--binary')
parser.add_argument('--save_path')

# training parameter
parser.add_argument('--strip', help ='strip punctuations', type= str)
parser.add_argument('--voc_size', default=30000, type = int, help=' word embedding dimension')
parser.add_argument('--max_len', default=40, type = int, help=' max length of single sentence')
parser.add_argument('--min_count', default = 3, type = int, help=' min freq count in word2vec')
parser.add_argument('--wndw', default=5, type=int, help= 'window size in word2vec')
parser.add_argument('--iter', default=16, type=int, help=' iteration in word2vec')

args = parser.parse_args()

# save as sentence
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
            X.append(sentence[1])
        else:
            sentence = line.strip()
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
def word2matrix(voc_size, data):
    tokenizer = Tokenizer(num_words=voc_size, filters="")
    print(len(tokenizer.word_counts))
    tokenizer.fit_on_texts(data)
    data = data[:200000]
    one_hot = tokenizer.texts_to_matrix(data)
    print("lexicon size = %i" % voc_size)
    print("loading %i data with length %i" % (one_hot.shape[0], one_hot.shape[1]))
    with open(args.binary + '.pkl', 'wb') as handler:
        pk.dump(tokenizer, handler)
    return one_hot

def buildModel(voc_size):
    print('=== building model ===')
    model = Sequential()
    # DNN
    model.add(Dense(1024, activation='relu', input_shape=(voc_size, )))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
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
    history = model.fit(X_train, Y_train, batch_size=256, epochs=10,
              callbacks=[checkpoint, early_stop],
              validation_data=(X_valid, Y_valid)
              )

    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig("bow.png")


def main():
    PATH = args.save_path
    if args.strip == 'True':
        STRIP = True 
    else:
        STRIP = False

    X_train, Y = load_data(args.train_data, True, STRIP)
    X_semi, __ = load_data(args.semi_data, False, STRIP)
    
    X = word2matrix(args.voc_size, X_train+X_semi)
    model = buildModel(args.voc_size)
    supervised_training(model, PATH, X, Y) 

if __name__ == "__main__":
    main()
