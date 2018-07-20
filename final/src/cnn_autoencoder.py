import numpy as np
import pandas as pd
import argparse
import random
import librosa.display
import matplotlib.pyplot as plt
from para import *
from utils import *
from mydataGen import *

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

def read_csv(data_path, mode='train'):
    print ('  Loading data from %s...'%data_path)
    data_csv = pd.read_csv(data_path).values
    train_y_fname = {}
    test_fname = []
    count = 0
    for line in data_csv:
        print('    loaded num: {}\r'.format(count), end='')
        #if count == 1000: break
        if mode == 'train':
            if(line[2]):
                train_y_fname[line[0]] = category.index(line[1])
                count += 1
        elif mode == 'semi':
            train_y_fname[line[0]] = category.index(line[1])
            count += 1
        else:
            test_fname.append(line[0])
            count += 1
    data = {}
    if mode == 'train':
        data['train_data'] = train_y_fname
        #print(self.data['train_data'][0].shape)
    elif mode == 'semi':
        data['train_data'] = train_y_fname
    else:
        data['test_data'] = np.array([test_fname])
    print(" ")
    return data

def split_data(data, fraction=0.1):
    length = len(data)
    points = int(length*fraction)
    # get random key for # of points
    val_key = []
    partition = {}
    train_key = []
    for i in range(points):
        val_key.append(random.choice(list(data.items()))[0])
    for key in data:
        if(key not in val_key):
            train_key.append(key)
    partition['train'] = train_key
    partition['validation'] = val_key
    return partition


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

    X = np.array(X)
    label = np.array(label)
    filename = np.array(filename)
    X.reshape(len(X), width ,height)
    X = X[:, :, :, np.newaxis]
    print('=== train data shape... {} ==='.format(X.shape))
    return X, label, filename

def getModel():
    # encoder
    input_img = Input(shape=(513, 439, 1), name='main_input')
    encode = Conv2D(16, (10, 4), strides=(2, 2), activation='relu')(input_img)
    encode = BatchNormalization()(encode)
    encode = Conv2D(16, (10, 4), activation='relu')(encode)
    encode = BatchNormalization()(encode)
    encode = MaxPooling2D(pool_size=(2, 2), padding = 'same')(encode)

    encode = Conv2D(32, (8, 4), strides=(2, 2), activation='relu')(encode)
    encode = BatchNormalization()(encode)
    encode = Conv2D(32, (8, 4), activation='relu')(encode)
    encode = BatchNormalization()(encode)
    encode = MaxPooling2D(pool_size=(2, 2), padding = 'same')(encode) 

    encode = Conv2D(64, (5, 4), activation='relu')(encode)
    encode = BatchNormalization()(encode)
    encode = Conv2D(128, (5, 4), activation='relu')(encode)
    encode = BatchNormalization()(encode)
    encode = MaxPooling2D(pool_size=(2, 2), padding = 'same')(encode) 

    encode = Conv2D(256, (3, 3), activation='relu')(encode)
    encode = BatchNormalization()(encode)
    encode = MaxPooling2D(pool_size=(2, 2), padding = 'same', name='encoded_output')(encode) 
    encode = Dropout(0.2)(encode)

    decode = Conv2DTranspose(128, (3, 3), activation='relu')(encode)
    decode = UpSampling2D(size=(2, 2))(decode)
    decode = BatchNormalization()(decode)

    decode = UpSampling2D(size=(2, 2))(decode)
    decode = BatchNormalization()(decode)
    decode = Conv2DTranspose(64, (3, 2), activation='relu')(decode)
    decode = BatchNormalization()(decode)
    decode = Conv2DTranspose(32, (4, 3), activation='relu')(decode)

    decode = UpSampling2D(size=(2, 2))(decode)
    decode = BatchNormalization()(decode)
    decode = Conv2DTranspose(32, (6, 2), strides=(2, 2), activation='relu')(decode)
    decode = BatchNormalization()(decode)
    decode = Conv2DTranspose(16, (6, 2), activation='relu')(decode)

    decode = UpSampling2D(size=(2, 2))(decode)
    decode = BatchNormalization()(decode)
    decode = Conv2DTranspose(16, (8, 3), strides=(2, 2), activation='relu')(decode)
    decode = BatchNormalization()(decode)
    decode = Conv2DTranspose(1, (8, 3), activation='linear')(decode)

    adam = Adam(lr=1e-5)
    AutoEncoder = Model(input=input_img, output = decode)
    AutoEncoder.compile(optimizer=adam, loss='mse')
    AutoEncoder.summary()
    return AutoEncoder

def prediction(model, X):
    res = model.predict(X, verbose=1)
    return X * (-80.0), res * (-80.0)

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
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()

        plt.subplot(2, PICTURE_NUM, i + PICTURE_NUM + 1)
        librosa.display.specshow(reconstructed[i], y_axis='log', x_axis='time', sr=sr)
        plt.title('recon #{}'.format(i))
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()

    plt.show()

def main(args):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.fraction
    set_session(tf.Session(config=config))
    # train
    if args.action == 'train':
        if args.load_path != 'None':
            print('=== load model from {}'.format(args.load_path))
            auto = load_model(args.load_path)
        else:
            auto = getModel()
        train_set = read_csv('data/train.csv', mode='semi')['train_data']
        partition = split_data(train_set)
        labels = train_set
        training_generator = DataGenerator(partition['train'], labels, **params)
        validation_generator = DataGenerator(partition['validation'], labels, **params)
        
        checkpoint = ModelCheckpoint(args.save_path, monitor='val_loss', verbose=1,
                                save_best_only=True, mode='min')
        early_stop = EarlyStopping(monitor='val_loss',
                                min_delta=0,
                                patience=10,
                                verbose=1, mode='min')        
        auto.fit_generator(training_generator, steps_per_epoch=len(partition['train'])//16, 
                                epochs=300, verbose=1, callbacks=[checkpoint, early_stop],
                                validation_data=validation_generator, validation_steps=len(partition['validation'])//16)
    # test
    elif args.action == 'test':
        auto = load_model(args.load_path)
        auto.summary()
        X, label, filename = getData('data/train.csv')
        X /= (-80.0)
        f = K.function([auto.layers[0].input, K.learning_phase()] , [Flatten()(auto.layers[18].output)])
        
        verified = 1

        for i in range(48):
            origin = X[i * 200: (i+1)* 200]
            layer_out = f([origin, 1])[0]
            print('split # {}/ 48 with shape{} \r'.format(i+1, layer_out.shape), end='')
            for j in range(200):
                if i * 200 + j > 3710:
                    verified = 0
                path = train_enc_dir + filename[i*200+j] + '.npy'
                np.save(path, [layer_out[j], label[i*200+j], verified])
    # plot
    else:
        X, label, filename = getData('data/train.csv')
        auto = load_model(args.load_path)
        origin = X[30:33] /(-80.0)
        origin, recon = prediction(auto, origin)
        plotImage(origin, recon)


if __name__ == "__main__":
    args = ArgumentParser()
    main(args)