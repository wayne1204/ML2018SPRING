import pandas as pd
import numpy as np
import csv
import math
import sys
import matplotlib.pyplot as plt
import scipy.misc as smp

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
	return filename

def scaling(filename):
    filename = filename.reshape(filename.shape[0],48, 48,1)
    filename = filename.astype('float64')
    filename /= 255
    return filename

# def plotting():
    

if __name__ == '__main__':
    num = int(sys.argv[1])
    training_set = parsingData('data/train.csv')
    training_label = parsingLabel('data/train.csv')
    training_set = scaling(training_set)
    data = training_set[num].reshape((48,48))
    img = smp.toimage(data)
    img.resize((240,240))
    img.show()
    print("picture #%i is class#%i" % (num, training_label[num]))
	
