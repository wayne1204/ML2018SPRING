from sys import argv
import csv
import numpy as np
from sklearn.decomposition import PCA
from MulticoreTSNE import MulticoreTSNE as TSNE
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.decomposition import TruncatedSVD
from utils import *
from para import *



def getdata():
	dm = DataManager()
	#dm.add_rawData('data/train.csv', mode='semi')
	#dm.preprocess()
	dm.load_data(train_enc_dir,mode='semi')
	train_data = dm.get_data('train_data')[0]
	semi_data = dm.get_data('semi_data')[0]
	train_label = dm.get_data('train_data')[1]
	semi_label = dm.get_data('semi_data')[1]
	train_name = dm.get_data('train_data')[2]
	semi_name = dm.get_data('semi_data')[2]
	return train_data, semi_data, train_label, semi_label, train_name, semi_name


def writecsv(trainname,newlabel,oldlabel):
	file = open('newtrain.csv','w')
	file.write('fname,label,manually_verified\n')
	similarity = 0
	for i in range(len(trainname)):
		file.write(str(trainname[i])+','+str(category[oldlabel[i]])+',')
		if oldlabel[i] == newlabel[i]:
			file.write(str(1)+'\n')
			similarity+=1
		else:
			file.write(str(0)+'\n')
	file.close()
	print('similarity: ', similarity/trainname.shape[0])


def main():

	train_data, semi_data, train_label, semi_label, train_name, semi_name = getdata()

	train_dm = []
	train_dnm = []
	train_lbm = []
	train_lbnm = []
	
	semi_dm = []
	semi_dnm = []
	semi_lbm = []
	semi_lbnm = []

	name_m = []
	name_nm = []


	for i in range(len(train_data)):
		if train_label[i] in music_category:
			train_dm.append(train_data[i])
			train_lbm.append(train_label[i])
			name_m.append(train_name[i])
		else:
			train_dnm.append(train_data[i])
			train_lbm.append(train_label[i])
			name_nm.append(train_name[i])

	for i in range(len(semi_data)):
		if semi_label[i] in music_category:
			semi_dm.append(semi_data[i])
			semi_lbm.append(semi_label[i])
			name_m.append(semi_name[i])
		else:
			semi_dnm.append(semi_data[i])
			semi_lbm.append(semi_label[i])
			name_nm.append(semi_name[i])


	name = np.array(name)
	data = np.array(data)
	semi_sp_data = np.array(semi_sp_data)
	data = data/-80.
	semi_sp_data = semi_sp_data/-80.
	print('name size: ', name.shape[0])
	print('data train size: ',data.shape[0]-semi_sp_data.shape[0])
	print('data semi size: ', semi_sp_data.shape[0])
	print('PCA...')
	#data = PCA(n_components=439,whiten=True,svd_solver="full",random_state=0).fit_transform(data)
	#SVD = TruncatedSVD(n_components=200, algorithm='arpack')
	#SVD.fit(data)
	#data = SVD.transform(data)
	#semi_sp_data = SVD.transform(semi_sp_data)

	print('data PCA size: ',data.shape)
	semi_unlabel = np.full(semi_label.shape[0],-1)
	label = np.concatenate((train_label,semi_unlabel),axis=0)
	label = label.astype('int')
	data = data.astype('float')
	print('label propagation...')	
	#print(label)
	#model = LabelPropagation(kernel='knn',n_neighbors=5,max_iter=10000,tol=0.001,n_jobs=-1)
	model = LabelSpreading(kernel='knn',gamma=20,alpha=0.2,n_neighbors=100,max_iter=100000,tol=0.001,n_jobs=20)
	model.fit(data,label)
	oursemi_label = model.predict(semi_sp_data)
	ourlabel = np.concatenate((train_label,oursemi_label),axis=0)
	csvlabel = np.concatenate((train_label,semi_label),axis=0)
	print(ourlabel)
	#print(csvlabel)
	print('data size: ',data.shape)
	print('ourlabel size: ', ourlabel.shape)
	print('csvlabel size: ', csvlabel.shape)
	writecsv(name,ourlabel,csvlabel)
	#csvlabel = train_label
	print('TSNE...')
	pic = TSNE(n_components=2,n_jobs=20).fit_transform(data)

	dicCSV = {}
	for i in range(pic.shape[0]):
		for j in range(41):
			if (int(csvlabel[i])==int(j)):
				if j in dicCSV:
					dicCSV[j].append([pic[i,0],pic[i,1]])
					#print(j,' ',np.array(dicCSV[j]).shape)
				else: 
					dicCSV[j]=[]
					dicCSV[j].append([pic[i,0],pic[i,1]])
	
	dicOUR = {}
	for i in range(pic.shape[0]):
		for j in range(41):
			if (int(ourlabel[i])==int(j)):
				if j in dicOUR:
					dicOUR[j].append([pic[i,0],pic[i,1]])
					#print(j,' ',np.array(dicOUR[j]).shape)
				else: 
					dicOUR[j]=[]
					dicOUR[j].append([pic[i,0],pic[i,1]])
	
	print('plot...')
	print('CSV genres...',len(dicCSV))
	fig1,axes=plt.subplots(2,2)
	colors = cm.rainbow(np.linspace(0, 1, int(len(dicCSV)/4)))
	for n,i in enumerate(dicCSV):
		for x in range(2):
			for y in range(2):
				if(int(n/(int(len(dicCSV)/4)))==(x*2+y)):
					temp = np.array(dicCSV[i])
					axes[x,y].scatter(temp[:,0],temp[:,1],color=colors[n%(int(len(dicCSV)/4))],s=5)
					#for k in range(temp.shape[0]):
						#axes[x,y].annotate(str(i),(temp[k,0],temp[k,1]))
						#axes[x,y].text(temp[k,0]*1.05,temp[k,1]*1.05,str(i),fontdict={'size': 7})
	fig1.savefig('TSNE_cluster/TSNEcsv.png')
	plt.close()
	
	print('OUR genres...',len(dicCSV))
	fig2,axes=plt.subplots(2,2)
	colors = cm.rainbow(np.linspace(0, 1, int(len(dicOUR)/4)))
	for n,i in enumerate(dicOUR):
		for x in range(2):
			for y in range(2):
				if(int(n/(int(len(dicOUR)/4)))==(x*2+y)):
					temp = np.array(dicOUR[i])
					axes[x,y].scatter(temp[:,0],temp[:,1],color=colors[n%(int(len(dicOUR)/4))],s=5)
					#for k in range(temp.shape[0]):
						#axes[x,y].annotate(str(i),(temp[k,0],temp[k,1]))
						#axes[x,y].text(temp[k,0]*1.05,temp[k,1]*1.05,str(i),fontdict={'size': 7})
	fig2.savefig('TSNE_cluster/TSNEour.png')
	plt.close()
	

if __name__ == '__main__':
	main()
