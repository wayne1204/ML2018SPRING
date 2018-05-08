import pandas as pd
import numpy as np
import csv
import math
import sys

from keras.models import Sequential, load_model
from keras.layers import Dense,Dropout, Flatten, Activation
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint

def parsingData(path):
	print('=== parsing file from %s ===' % path)
	filename = []
	number = -1

	text = open(path, 'r', encoding = 'big5')
	rows = csv.reader(text, delimiter = ',')

	for r in rows:
		if number != -1:
			n_row = [float(t) for t in r[1].split(' ')]
			filename.append(n_row)
		number += 1	

	filename = np.array(filename)
	return filename

def parsingLabel(path):
	print('=== generating label ===')
	filename = pd.read_csv(path, usecols= ['label'] )
	filename = np.array(filename)
	filename = np_utils.to_categorical(filename, 7)
	return filename

def shuffle_split(X_all, Y_all, percentage):
	print('=== shuffling... ===')
	all_size = X_all.shape[0]
	randomize = np.arange(all_size)
	X_all, Y_all = X_all[randomize], Y_all[randomize]
	valid_size = int(math.floor(all_size * percentage))
	X_train, Y_train = X_all[0:valid_size], Y_all[0:valid_size]
	X_valid, Y_valid = X_all[valid_size:], Y_all[valid_size:]
	return X_train, Y_train, X_valid, Y_valid

def scaling(filename):
    filename = filename.reshape(filename.shape[0],48, 48,1)
    filename = filename.astype('float64')
    filename /= 255
    return filename


def training(training_set, training_label, validation_set, validation_label):
	print('=== building CNN model... ===')
	model = Sequential()

	model.add(Convolution2D(32, (3,3), activation='selu',input_shape=(48,48,1)))
	model.add(BatchNormalization())
	model.add(Convolution2D(32, (3,3), activation='selu',input_shape=(48,48,1)))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

	model.add(Dropout(0.2))

	model.add(Convolution2D(64, (3,3), activation='selu'))
	model.add(BatchNormalization())
	model.add(Convolution2D(64, (3,3), activation='selu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

	model.add(Dropout(0.2))

	model.add(Convolution2D(128, (3,3), activation='selu'))
	model.add(BatchNormalization())
	model.add(Convolution2D(256, (3,3), activation='selu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

	model.add(Dropout(0.3))

	model.add(Flatten())
	model.add(Dense(1024, activation='selu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))
	model.add(Dense(512, activation='selu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Dense(7, activation='softmax'))

	# Compile model
	datagen = ImageDataGenerator(
	rotation_range=10,
	width_shift_range=0.1,
	height_shift_range=0.1,
	horizontal_flip =True, 
	)

	datagen.fit(training_set)
	print(training_set.shape)
	model.compile(loss='categorical_crossentropy',
				optimizer='Adam',
				metrics=['accuracy'])

	# model.fit(training_set, training_label, 
	#           batch_size=32, epochs=2, verbose=2, validation_split=0.1) 
	cb = EarlyStopping(monitor='val_loss',
						min_delta=0,
						patience=10,
						verbose=1, mode='auto')
	model.fit_generator(datagen.flow(training_set, training_label, batch_size=256),
					steps_per_epoch=len(training_set) / 256 * 8, epochs=100, verbose=2,
					callbacks=[cb], validation_data=(validation_set, validation_label))
	model.save('my_model.h5')
	model.summary()


if __name__ == '__main__':
	filename = sys.argv[1]
	training_set = parsingData(filename)
	training_label = parsingLabel(filename)
	training_set = scaling(training_set)
	# testing_set = scaling(testing_set)
	
	training_set, training_label, validation_set, validation_label = shuffle_split(training_set, training_label, 0.9)
	training(training_set, training_label, validation_set, validation_label)

