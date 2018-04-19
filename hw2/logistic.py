import pandas as pd
import numpy as np
import math
import csv


def readingData(path):
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

def readingLabel(path):
	data = pd.read_csv(path, header=None)
	data = np.array(data)
	return data

def standardize(training_set, testing_set):
	print('standardizing...')
	mu = np.mean(training_set, axis = 0, dtype=np.float64)
	sigma = np.std(training_set, axis = 0, dtype=np.float64)
	mu_1 = np.tile(mu, (training_set.shape[0], 1))
	sigma_1 = np.tile(sigma, (training_set.shape[0], 1))
	mu_2 = np.tile(mu, (testing_set.shape[0], 1))
	sigma_2 = np.tile(sigma, (testing_set.shape[0], 1))
	training_set = (training_set - mu_1) / sigma_1
	testing_set = (testing_set - mu_2) / sigma_2
	np.save('mu.npy',mu)
	np.save('sigma.npy',sigma)
	return training_set, testing_set

def shuffle_split(X_all, Y_all, percentage):
	print('shuffling...')
	all_size = X_all.shape[0]
	randomize = np.arange(all_size)
	X_all, Y_all = X_all[randomize], Y_all[randomize]
	valid_size = int(math.floor(all_size * percentage))
	X_train, Y_train = X_all[0:valid_size], Y_all[0:valid_size]
	X_valid, Y_valid = X_all[valid_size:], Y_all[valid_size:]
	return X_train, Y_train, X_valid, Y_valid

def sigmoid(z):
	y = 1 / ( 1 + np.exp(-z))
	return np.clip(y, 1e-8, 1-(1e-8))

def validationScore(X_valid,Y_valid, w, b):
	sigmoid_valid = sigmoid(np.dot(X_valid, w) + b)
	result = np.around(sigmoid_valid) - Y_valid
	result = np.abs(result)
	correct = 1 - float(np.sum(result)) / X_valid.shape[0]
	return correct

def prediction(testing_set, w, b):
	test = np.dot(testing_set, w)
	sigmoid_test = sigmoid(test + b)
	sigmoid_test = np.around(sigmoid_test)
	ans = []
	for i in range(len(sigmoid_test)):
		ans.append([i+1])
		ans[i].append(int(sigmoid_test[i][0]))

	filename = 'predict.csv'
	text = open(filename, "w+")
	s = csv.writer(text,delimiter=',',lineterminator='\n')
	s.writerow(["id","label"])
	for i in range(len(ans)):
	    s.writerow(ans[i]) 
	text.close()

def training(X_train, Y_train):
	w = np.ones((len(X_train[0]),1))
	x_t = X_train.transpose()
	gra_sum = np.ones((len(X_train[0]),1))
	b = np.zeros((1,1))
	gra_sum_b = b;
	l_rate = 0.01
	b_rate = 1
	epoch_num = 15000
	batch_size = len(X_train)
	batch_num = len(X_train)//batch_size
	
	beta1 = 0.9
	beta2 = 0.999
	e = math.pow(10, -8)
	first_moment_w = 0
	second_moment_w = 0
	first_moment_b = 0
	second_moment_b = 0
	regulation = 0
	total_loss = 0

	for epoch in range(epoch_num):

		total_loss = 0
		for idx in range(batch_num):
			X = training_set[idx*batch_size:(idx+1) * batch_size]
			Y = training_label[idx*batch_size:(idx+1) * batch_size]
			hypo = np.dot(X,w)
			sigmoid_y = sigmoid(hypo + b) 
			loss = sigmoid_y - Y
			cost = np.dot(np.squeeze(Y), np.log(sigmoid_y)) + np.dot((1 - np.squeeze(Y)), np.log(1 - sigmoid_y))
			total_loss += -1*np.sum(cost)/len(training_set)
			gra = np.dot(np.transpose(X), loss) + regulation*w 
			gra_b = np.sum(loss) + regulation*b 

			first_moment_w = beta1 * first_moment_w + (1 - beta1) * gra
			second_moment_w = beta2 * second_moment_w + (1 - beta2) * gra**2
			first_moment_b = beta1 * first_moment_b + (1 - beta1) * gra_b
			second_moment_b = beta2 * second_moment_b + (1 - beta2) * gra_b**2

			first_moment_wb = first_moment_w / ( 1 - math.pow(beta1, epoch+1))
			second_moment_wb = second_moment_w / ( 1 - math.pow(beta2, epoch+1))
			first_moment_bb = first_moment_b / ( 1 - math.pow(beta1, epoch+1))
			second_moment_bb = second_moment_b / ( 1 - math.pow(beta2, epoch+1))
			w = w - l_rate * first_moment_wb / (np.sqrt(second_moment_wb) + e)	
			b = b - b_rate * first_moment_bb / (np.sqrt(second_moment_bb) + e)	

		if(epoch % 100 == 0):
			print('==============[epoch  %d]============== ' % ( epoch))
			print('Total Lost %f | Score: %f | V_Score: %f' % ( total_loss,validationScore(X_train,Y_train, w, b), validationScore(X_valid,Y_valid, w, b)))

	np.save('model.npy',w)
	np.save('bias_term.npy',b)
	return w,b

if __name__ == '__main__':
	training_set = readingData('data/train_X')
	testing_set = readingData('data/test_X')
	training_label = readingLabel('data/train_Y')
	training_set, testing_set = standardize(training_set, testing_set)	
	X_train, Y_train, X_valid, Y_valid = shuffle_split(training_set, training_label, 0.9)
	w,b = training(X_train, Y_train)
