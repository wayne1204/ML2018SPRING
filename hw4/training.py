import skimage
import numpy as np
import csv
import time
import sys
from skimage import data
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


def parsingTestCase(path):
	print('=== parsing file from %s ===' % path)
	filename = []
	index = -1

	text = open(path, 'r', encoding = 'big5')
	rows = csv.reader(text, delimiter = ',')

	for r in rows:
		if index != -1:
			filename.append([])
			for i in range(len(r)):
				filename[index].append( int(r[i]) )

		index += 1	

	filename = np.array(filename)
	return filename


def loadImage(path, num):
	imageArray = np.load(path)
	print('orginal dimension ', imageArray.shape)

	pca = PCA(n_components=num, copy=False, whiten=True, svd_solver='full')
	newData = pca.fit_transform(imageArray)  
	print('reduced dimension ' , newData.shape)
	return newData

def clustering(data):
	kmeans = KMeans(n_clusters=2, random_state=100).fit(data)
	count = 0
	for i in range(len(kmeans.labels_)):
		if kmeans.labels_[i] == 0:
			count += 1

	print('class 0: %i | class 1: %i' % (count, len(kmeans.labels_)-count))
	return kmeans.labels_

def tsneReduce(reduced_data):
	time_start = time.time()
	tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=300)
	tsne_results = tsne.fit_transform(reduced_data)
	time_end = time.time()
	print('total used time: {}'.format(time_start - time_end))
	return tsne_results

def prediction(Label, test):
	result = []
	one_c = 0
	zero_c = 0
	for i in range(len(test)):
		result.append([])
		result[i].append(i)
		index1 = test[i][1]
		index2 = test[i][2]
		if Label[index1] != Label[index2]:
			result[i].append(0)
			zero_c += 1
		else:
			result[i].append(1)
			one_c += 1
	print('zero count: %i| one count: %i' % (zero_c, one_c))
	return result

def outputAns(result, file):
	text = open(file, "w+")
	s = csv.writer(text, delimiter=',', lineterminator='\n')
	s.writerow(["id","Ans"])
	for i in range(len(result)):
		s.writerow(result[i]) 
	text.close()
	

if __name__ == '__main__':
	reducedData = loadImage(sys.argv[1], 400)
	# Label = tsneReduce(reducedData)
	Label = clustering(reducedData)
	TestCase = parsingTestCase(sys.argv[2])
	result = prediction(Label, TestCase)
	outputAns(result, sys.argv[3])
	
