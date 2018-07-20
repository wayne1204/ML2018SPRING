import numpy as np
import pandas as pd
import argparse
import random
import librosa.display
import matplotlib.pyplot as plt
from para import *
from utils import *
from dataGen import *

from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import ZeroPadding2D, UpSampling2D, Reshape, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

def ArgumentParser():
    parser = argparse.ArgumentParser(description='=== spectrum autoencoder ===')
    parser.add_argument('action',choices=['train', 'test', 'plot'], default='train')
    parser.add_argument('--load_path', default='None', type=str)
    parser.add_argument('--save_path', default='model/model_2.h5', type=str)
    parser.add_argument('--fraction', default=0.5, type=float)
    return parser.parse_args()

def shuffleSplit(data, fraction = 0.9):
    print('    Shuffling data...')
    indexing = np.arange(len(data))
    np.random.shuffle(indexing)
    data = data[indexing]
    train_data = data[:8800]
    test_data = data[8800:]
    print(train_data.shape)
    print(test_data.shape)
    return train_data, test_data

def normalization(data, done = True):
    print('    Normalizing data...')
    if not done:
        mean = np.mean(data, axis=0)
        sigma = np.std(data, axis=0)
        np.save('model/mean.npy', mean)
        np.save('model/sigma.npy', sigma)
    else:
        mean = np.load('model/mean.npy')
        sigma = np.load('model/sigma.npy')
    return (data - mean) / (sigma + 1e-5)


def getData(data_path):
    X, label, filename = [], [], []
    dm = DataManager()
    dm.load_data(train_spectrum_dir, mode='semi')
    train_data = dm.get_data('train_data')[0]
    semi_data = dm.get_data('semi_data')[0]
    train_label = dm.get_data('train_data')[1]
    semi_label = dm.get_data('semi_data')[1]
    train_data_name = dm.get_data('train_data')[2]
    semi_data_name  = dm.get_data('semi_data')[2]

    width = train_data[0].shape[0]
    height = train_data[0].shape[1]
    print(len(train_data))
    for i in range(len(train_data)):
        X.append(train_data[i])
        label.append(train_label[i])
        filename.append(train_data_name[i])
    for i in range(len(semi_data)):
        X.append(semi_data[i])
        label.append(semi_label[i])
        filename.append(semi_data_name[i])

    X = np.array(X, dtype='float64')
    label = np.array(label)
    filename = np.array(filename)
    X.reshape(len(X), width ,height)
    X = X[:, :, :, np.newaxis]
    print('=== train data shape... {} ==='.format(X.shape))
    return X, label, filename


def loadBatchData(X, batch_size):
    loop_count = len(X) // batch_size
    while True:
        i = random.randint(0, loop_count)
        batch = X[i * batch_size: (i+1) * batch_size]
        yield batch, batch


def getModel():
    # encoder
    input_img = Input(shape=(513, 439, 1))
    encode = Conv2D(16, (10, 4), strides=(2, 2), activation='relu')(input_img)
    # encode = BatchNormalization()(encode)
    encode = Conv2D(16, (10, 4), activation='relu')(encode)
    # encode = BatchNormalization()(encode)
    encode = MaxPooling2D(pool_size=(2, 2), padding = 'same')(encode)

    encode = Conv2D(32, (8, 4), strides=(2, 2), activation='relu')(encode)
    # encode = BatchNormalization()(encode)
    encode = Conv2D(32, (8, 4), activation='relu')(encode)
    # encode = BatchNormalization()(encode)
    encode = MaxPooling2D(pool_size=(2, 2), padding = 'same')(encode) 

    encode = Conv2D(64, (5, 4), activation='relu')(encode)
    # encode = BatchNormalization()(encode)
    encode = Conv2D(128, (5, 4), activation='relu')(encode)
    # encode = BatchNormalization()(encode)
    encode = MaxPooling2D(pool_size=(2, 2), padding = 'same')(encode) 

    encode = Conv2D(256, (3, 3), activation='relu')(encode)
    # encode = BatchNormalization()(encode)
    encode = MaxPooling2D(pool_size=(2, 2), padding = 'same')(encode) 
    encode = Dropout(0.2)(encode)


    decode = Conv2DTranspose(128, (3, 3), activation='relu')(encode)
    decode = UpSampling2D(size=(2, 2))(decode)
    # decode = BatchNormalization()(decode)

    decode = UpSampling2D(size=(2, 2))(decode)
    # decode = BatchNormalization()(decode)
    decode = Conv2DTranspose(64, (3, 2), activation='relu')(decode)
    # decode = BatchNormalization()(decode)
    decode = Conv2DTranspose(32, (4, 3), activation='relu')(decode)

    decode = UpSampling2D(size=(2, 2))(decode)
    # decode = BatchNormalization()(decode)
    decode = Conv2DTranspose(32, (6, 2), strides=(2, 2), activation='relu')(decode)
    # decode = BatchNormalization()(decode)
    decode = Conv2DTranspose(16, (6, 2), activation='relu')(decode)

    decode = UpSampling2D(size=(2, 2))(decode)
    # decode = BatchNormalization()(decode)
    decode = Conv2DTranspose(16, (8, 3), strides=(2, 2), activation='tanh')(decode)
    # decode = BatchNormalization()(decode)
    decode = Conv2DTranspose(1, (8, 3), activation='linear')(decode)

    adam = Adam(lr=1e-5, clipnorm=1.0)
    AutoEncoder = Model(input=input_img, output = decode)
    AutoEncoder.compile(optimizer=adam, loss='mse')
    AutoEncoder.summary()
    return AutoEncoder

def trainModel(save_path, AutoEncoder, train, valid):
    checkpoint = ModelCheckpoint(save_path, monitor='val_loss', verbose=1,
                                save_best_only=True, mode='min')
    early_stop = EarlyStopping(monitor='val_loss',
                               min_delta=0,
                               patience=10,
                               verbose=1, mode='min')
    
    # AutoEncoder.fit(train, train, epochs=30, batch_size=16, shuffle=True, verbose=1, 
                    # validation_split=0.1, callbacks=[early_stop, checkpoint])
    AutoEncoder.fit_generator(loadBatchData(train, 16), steps_per_epoch=len(train)//16, 
                              epochs=100, verbose=1, callbacks=[checkpoint, early_stop],
                              validation_data=[valid, valid])

def prediction(model, X):
    res = model.predict(X, verbose=1)
    mean = np.load('model/mean.npy')
    sigma = np.load('model/sigma.npy')
    res = res * sigma + mean
    X = X *sigma + mean
    return X, res

def plotImage(origin, reconstructed):
    PICTURE_NUM = len(origin)
    origin = origin.reshape(PICTURE_NUM, 513, 439)
    reconstructed = reconstructed.reshape(PICTURE_NUM, 513, 439)
    shape = origin[0].shape
    print(shape)

    for i in range(PICTURE_NUM):
        plt.subplot(2, PICTURE_NUM, i + 1)
        librosa.display.specshow(origin[i], y_axis='log', x_axis='time', sr=sr)
        plt.title('origin #{}'.format(i))
        # plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()

        plt.subplot(2, PICTURE_NUM, i + PICTURE_NUM + 1)
        librosa.display.specshow(reconstructed[i], y_axis='log', x_axis='time', sr=sr)
        plt.title('recon #{}'.format(i))
        # plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
    plt.show()

def main(args):

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.fraction
    set_session(tf.Session(config=config))
    X, label, filename = getData('data/train.csv')
    # train
    if args.action == 'train':
        X = normalization(X, True)
        train_X, valid_X = shuffleSplit(X)

        if args.load_path != 'None':
            print('=== load model from {}'.format(args.load_path))
            auto = load_model(args.load_path)
        else:
            auto = getModel()
        trainModel(args.save_path, auto, train_X, valid_X)
        
        # for j in range(10):
        #     for i in range(10):
        #         print('=== split #{}'.format(i+1))
        #         trainModel(args.save_path, auto, X[i*1000: (i+1)*1000], X[i*1000: (i+1)*1000])
    # test
    elif args.action == 'test':
        auto = load_model(args.load_path)
        auto.summary()
        X = normalization(X, True)
        f = K.function([auto.layers[0].input, K.learning_phase()] , [Flatten()(auto.layers[27].output)])
        
        verified = 1
        for i in range(50):
            print('split #', i)
            origin = X[i * 200: (i+1)* 200]
            layer_out = f([origin, 1])[0]
            print(layer_out.shape)
            for j in range(200):
                if i * 200 + j < 3710:
                    verified = 0
                path = train_enc_dir + filename[i*200+j] + '.wav.npy'
                np.save(path, [layer_out[j], label[i*200+j], verified])
    # plot
    else:
        X = normalization(X, True)
        auto = load_model(args.load_path)
        origin = X[50:54]
        origin, recon = prediction(auto, origin)
        plotImage(origin, recon)


if __name__ == "__main__":
    args = ArgumentParser()
    main(args)