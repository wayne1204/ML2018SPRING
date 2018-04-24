import pandas as pd
import numpy as np
import csv
import math
import itertools
import matplotlib.pyplot as plt
from scipy.misc import imsave

from keras import backend as K
from keras.models import Sequential, load_model
from keras.utils import np_utils



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


def prediction(testing_set):
    model = load_model('model/my_model.h5')    
    result = model.predict(testing_set, verbose=1)
    result = np.argmax(result, axis=1)
    return result

def outputFile(result):
    ans = []
    for i in range(len(result)):
        ans.append([i])
        ans[i].append(int(result[i]))

    filename = 'predict.csv'
    text = open(filename, "w+")
    s = csv.writer(text, delimiter=',', lineterminator='\n')
    s.writerow(["id","label"])
    for i in range(len(ans)):
        s.writerow(ans[i]) 
    text.close()


def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())

if __name__ == '__main__':
    training_set = parsingData('data/train.csv')
    training_label = parsingLabel('data/train.csv')
    testing_set = parsingData('data/test.csv')
    training_set = scaling(training_set)
    testing_set = scaling(testing_set)

    model = load_model('model/my_model3.h5')
    input_img = training_set[0]
    # input_img = model.input
    img_width, img_height = 48, 48
    kept_filters = []
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    
    
    for filter_index in range(32):
        print('Processing filter %d' % filter_index)

        # we build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        
        layer_output = layer_dict['conv2d_1'].output
        print(type(layer_output))
        loss = K.mean(layer_output[:, :, :, filter_index])
        print(type(loss))
        # we compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)
        print(grads)
        print(type(grads))
        print(type(input_img))
        # normalization trick: we normalize the gradient
        # grads = normalize(grads)
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

        # this function returns the loss and grads given the input picture
        iterataion = K.function([input_img], [loss, grads])

        # step size for gradient ascent
        step = 1.

        # we start from a gray image with some random noise
        if K.image_data_format() == 'channels_first':
            input_img_data = np.random.random((1, 1, 48, 48))
        else:
            input_img_data = np.random.random((1, 48, 48, 1))
        input_img_data = (input_img_data - 0.5) * 20 + 128

        # we run gradient ascent for 20 steps
        for i in range(20):
            loss_value, grads_value = iterataion([input_img_data])
            input_img_data += grads_value * step

            print('Current loss value:', loss_value)
            if loss_value <= 0.:
                # some filters get stuck to 0, we can skip them
                break

        # decode the resulting input image
        if loss_value > 0:
            img = deprocess_image(input_img_data[0])
            kept_filters.append((img, loss_value))

    # we will stich the best 64 filters on a 8 x 8 grid.
    n = 6

    # the filters that have the highest loss are assumed to be better-looking.
    # we will only keep the top 64 filters.
    kept_filters.sort(key=lambda x: x[1], reverse=True)
    kept_filters = kept_filters[:n * n]

    # build a black picture with enough space for
    # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
    margin = 5
    width = n * img_width + (n - 1) * margin
    height = n * img_height + (n - 1) * margin
    stitched_filters = np.zeros((width, height, 3))

    # fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            img, loss = kept_filters[i * n + j]
            stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                            (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

    # save the result to disk
    imsave('stitched_filters_%dx%d.png' % (n, n), stitched_filters)
    result = prediction(testing_set)
    outputFile(result)