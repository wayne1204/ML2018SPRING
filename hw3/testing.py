import pandas as pd
import numpy as np
import csv
import math
import sys

from keras.models import Sequential, load_model
import itertools
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

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

def scaling(filename):
	filename = filename.reshape(filename.shape[0],48, 48,1)
	filename = filename.astype('float64')
	filename /= 255
	return filename

def single_prediction(model, testing_set):
    print('=== load model from %s ===' % model)
    model = load_model(model)
    model.summary()
    result = model.predict(testing_set, verbose=1)
    return result

def outputAns(result, file):
    scalar = np.argmax(result, axis = 1)
    ans = []
    for i in range(len(scalar)):
        ans.append([i])
        ans[i].append(int(scalar[i]))

    text = open(file, "w+")
    s = csv.writer(text, delimiter=',', lineterminator='\n')
    s.writerow(["id","label"])
    for i in range(len(ans)):
	    s.writerow(ans[i]) 


if __name__ == '__main__':
    testing_set = parsingData(sys.argv[1])
    testing_set = scaling(testing_set)
    result = single_prediction('my_ensemble_model.h5', testing_set)
    outputAns(result, sys.argv[2])
    