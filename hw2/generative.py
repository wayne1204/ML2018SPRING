import pandas as pd
import numpy as np
import math
import csv
import sys

def readingData(path):
	data = pd.read_csv(path)
	data = np.array(data)
	for i in [0,10,78,79,80]:
		new_col = data[:,i].reshape( len(data),1)
		data = np.concatenate( (data, new_col**2), axis=1 )
		data = np.concatenate( (data, new_col**3), axis=1 )
		data = np.concatenate( (data, new_col**4), axis=1 )
		data = np.concatenate( (data, new_col**5), axis=1 )
	return data

def readingLabel(path):
	data = pd.read_csv(path, header=None)
	data = np.array(data)
	return data

def standardize(training_set, testing_set):
	mu = np.mean(training_set, axis = 0, dtype=np.float64)
	sigma = np.std(training_set, axis = 0, dtype=np.float64)
	mu_1 = np.tile(mu, (training_set.shape[0], 1))
	sigma_1 = np.tile(sigma, (training_set.shape[0], 1))
	mu_2 = np.tile(mu, (testing_set.shape[0], 1))
	sigma_2 = np.tile(sigma, (testing_set.shape[0], 1))
	training_set = (training_set - mu_1) / sigma_1
	testing_set = (testing_set - mu_2) / sigma_2
	return training_set, testing_set

def shuffle_split(X_all, Y_all, percentage):
	all_size = X_all.shape[0]
	randomize = np.arange(all_size)
	X_all, Y_all = X_all[randomize], Y_all[randomize]
	valid_size = int(math.floor(all_size * percentage))
	X_train, Y_train = X_all[0:valid_size], Y_all[0:valid_size]
	X_valid, Y_valid = X_all[valid_size:], Y_all[valid_size:]
	return X_train, Y_train, X_valid, Y_valid

def mean(X_train, Y_train):
	cnt1 = 0
	cnt2 = 0
	mu1 = np.zeros((X_train.shape[1], ))
	mu2 = np.zeros((X_train.shape[1], ))
	for i in range(len(X_train)):
		if Y_train[i] == 1:
			mu1 += X_train[i]
			cnt1 += 1
		else:
			mu2 += X_train[i]
			cnt2 += 1
	mu1 /= cnt1
	mu2 /= cnt2
	return mu1, mu2, cnt1, cnt2

def sigma(X_train, Y_train, mu1, mu2, cnt1, cnt2):
	dim = X_train.shape[1]
	sigma1 = np.zeros((dim, dim))
	sigma2 = np.zeros((dim, dim))
	print(X_train[0].shape, mu1.shape)
	for i in range(len(X_train)):
		if Y_train[i] == 1:
			sigma1 += np.dot(np.transpose([X_train[i] - mu1]),[(X_train[i] - mu1)])
		else:
			sigma2 += np.dot(np.transpose([X_train[i] - mu2]),[(X_train[i] - mu2)])
	sigma1 /= cnt1
	sigma2 /= cnt2
	share_sigma = (float(cnt1) / len(X_train)) * sigma1 + (float(cnt2) / len(X_train)) * sigma2
	return share_sigma


def sigmoid(z):
	y = 1 / ( 1 + np.exp(-z))
	return np.clip(y, 1e-8, 1-(1e-8))

def genModel(share_sigma, mu1, mu2, N1, N2):
	sigma_inv = np.linalg.inv(share_sigma)
	w = np.dot( (mu1-mu2), sigma_inv )
	b = (-0.5) * np.dot(np.dot([mu1], sigma_inv), mu1) + 0.5 * np.dot(np.dot([mu2], sigma_inv), mu2) + math.log(N1/N2)
	return w,b

def correctness(X_valid, Y_valid, w ,b):
	x = X_valid.T
	a = np.dot(w,x) + b
	y = np.around(sigmoid(a))
	y = y.reshape(len(y), 1)

	result = y - Y_valid
	result = np.abs(result)
	correct = 1 - float(np.sum(result)) / X_valid.shape[0]
	return correct
	
def prediction(testing_set, w, b):
	x = testing_set.T
	a = np.dot(w,x) + b
	y = np.around(sigmoid(a))

	ans = []
	for i in range(len(testing_set)):
		ans.append([i+1])
		ans[i].append(int(y[i]))

	filename = sys.argv[4]
	text = open(filename, "w+")
	s = csv.writer(text,delimiter=',',lineterminator='\n')
	s.writerow(["id","label"])
	for i in range(len(ans)):
	    s.writerow(ans[i]) 
	text.close()

if __name__ == '__main__':
	training_set = readingData(sys.argv[1])
	training_label = readingLabel(sys.argv[2])
	testing_set = readingData(sys.argv[3])
	# training_set, testing_set = standardize(training_set, testing_set)	
	X_train, Y_train, X_valid, Y_valid = shuffle_split(training_set, training_label, 0.8)
	mu1, mu2, count1, count2 = mean(X_train, Y_train)
	share_sigma = sigma(X_train, Y_train, mu1, mu2, count1, count2)	
	w,b = genModel(share_sigma, mu1, mu2, count1, count2)
	print('Score = %f | Validation_Score = %f' % (correctness(X_train, Y_train, w, b), 
		correctness(X_valid, Y_valid, w, b)) )
	prediction(testing_set, w, b)
	

