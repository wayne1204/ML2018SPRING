import pandas as pd
import numpy as np
import math
import csv
import sys


def readingData(path):
	print('reading Data...')
	data = pd.read_csv(path)
	data = np.array(data)
	for i in [0,78,79,80]:
		new_col = data[:,i].reshape( len(data),1)
		data = np.concatenate( (data, data**2), axis=1 )
		data = np.concatenate( (data, new_col**3), axis=1 )
		data = np.concatenate( (data, new_col**4), axis=1 )
		data = np.concatenate( (data, new_col**5), axis=1 )
	data = np.delete(data,10 ,1)
	for i in range(16):
		data = np.delete(data,27 ,1)
	return data

def standardize(testing_set):
	print('standardizing...')
	mu = np.load('mu.npy')
	sigma = np.load('sigma.npy')
	mu_2 = np.tile(mu, (testing_set.shape[0], 1))
	sigma_2 = np.tile(sigma, (testing_set.shape[0], 1))
	testing_set = (testing_set - mu_2) / sigma_2
	return testing_set

def loadingModel():
	print('loadingModel...')
	weight = np.load('model_best.npy')
	bias = np.load('bias_term.npy')
	return weight, bias

def sigmoid(z):
	y = 1 / ( 1 + np.exp(-z))
	return np.clip(y, 1e-8, 1-(1e-8))
	
def prediction(testing_set, w, b):
	test = np.dot(testing_set, w)
	sigmoid_test = sigmoid(test + b)
	sigmoid_test = np.around(sigmoid_test)
	ans = []
	for i in range(len(sigmoid_test)):
		ans.append([i+1])
		ans[i].append(int(sigmoid_test[i][0]))

	filename = sys.argv[2]
	text = open(filename, "w+")
	s = csv.writer(text,delimiter=',',lineterminator='\n')
	s.writerow(["id","label"])
	for i in range(len(ans)):
	    s.writerow(ans[i]) 
	text.close()


if __name__ == '__main__':
	testing_set = readingData(sys.argv[1])
	testing_set = standardize(testing_set)	
	w, b = loadingModel()
	prediction(testing_set, w, b)


