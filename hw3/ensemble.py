import pandas as pd
import numpy as np
import csv
import math
import time

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

def outputRawFile(result):
    ans = []
    for i in range(len(result)):
        ans.append([i])
        for j in range(7):
            ans[i].append(float(result[i][j]))

    filename = 'ensemble.csv'
    text = open(filename, "w+")
    s = csv.writer(text, delimiter=',', lineterminator='\n')
    s.writerow(["id","label"])
    for i in range(len(ans)):
	    s.writerow(ans[i]) 

def outputAns(result):
    scalar = np.argmax(result, axis = 1)
    ans = []
    for i in range(len(scalar)):
        ans.append([i])
        ans[i].append(int(scalar[i]))

    filename = 'ans.csv'
    text = open(filename, "w+")
    s = csv.writer(text, delimiter=',', lineterminator='\n')
    s.writerow(["id","label"])
    for i in range(len(ans)):
	    s.writerow(ans[i]) 


if __name__ == '__main__':
    testing_set = parsingData('data/test.csv')
    testing_set = scaling(testing_set)
    result1 = single_prediction('model/my_model3.h5', testing_set)
    result2 = single_prediction('model/my_model7.h5', testing_set)
    result3 = single_prediction('model/my_model14.h5', testing_set)
    result4 = single_prediction('model/my_model15.h5', testing_set)
    result5 = single_prediction('model/my_model16.h5', testing_set)
    result = result1 + result2 + result3 + result4 + result5 
    outputRawFile(result)
    outputAns(result)
    