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
import warnings
from sklearn.svm import SVC,LinearSVC


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
	file = open('new_csv/newtrain.csv','w')
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
	print('num: ', similarity)
	print('similarity: ', similarity/trainname.shape[0])


def main():

	t_d, s_d, t_l, s_l, t_n, s_n = getdata()
	d1, n1, our1, csv1 = labelspread(t_d, s_d, t_l, s_l, t_n, s_n, musical, 'musical')
	d2, n2, our2, csv2 = labelspread(t_d, s_d, t_l, s_l, t_n, s_n, strings, 'strings')
	d3, n3, our3, csv3 = labelspread(t_d, s_d, t_l, s_l, t_n, s_n, percussion, 'percussion')
	d4, n4, our4, csv4 = labelspread(t_d, s_d, t_l, s_l, t_n, s_n, creature , 'creature')
	d5, n5, our5, csv5 = labelspread(t_d, s_d, t_l, s_l, t_n, s_n, non_human, 'non_human')

	#tsne(d1, our1, csv1, 'strings')
	#tsne(d2, our2, csv2, 'musical')
	#tsne(d3, our3, csv3, 'percussion')
	#tsne(d4, our4, csv4, 'creature')
	#tsne(d5, our5, csv5, 'non_human')
	name = np.concatenate((n1,n2,n3,n4,n5),axis=0)
	ourlabel = np.concatenate((our1,our2,our3,our4,our5),axis=0)
	csvlabel = np.concatenate((csv1,csv2,csv3,csv4,csv5),axis=0)

	writecsv(name,ourlabel,csvlabel)
	
	

def labelspread(train_data, semi_data, train_label, semi_label, train_name, semi_name, lib, libname):
	print("===========================")
	train_d = []
	train_l = []
	semi_d = []
	semi_l = []
	name = []

	for i in range(len(train_data)):
		if train_label[i] in lib:
			train_d.append(train_data[i])
			train_l.append(train_label[i])
			name.append(train_name[i])

	for i in range(len(semi_data)):
		if semi_label[i] in lib:
			semi_d.append(semi_data[i])
			semi_l.append(semi_label[i])
			name.append(semi_name[i])

	train_d = np.array(train_d)/-80.
	train_l = np.array(train_l)
	semi_d = np.array(semi_d)/-80.
	semi_l = np.array(semi_l)
	name = np.array(name)

	print(libname,' all num: ', train_d.shape[0]+semi_d.shape[0])
	print(libname,' train num: ', train_d.shape[0])
	print(libname,' ratio: ', train_d.shape[0]/(train_d.shape[0]+semi_d.shape[0]))
	#print('PCA...')
	#data = PCA(n_components=439,whiten=True,svd_solver="full",random_state=0).fit_transform(data)
	#semi_sp_data = SVD.transform(semi_sp_data)
	#print('data PCA size: ',data.shape)
	semi_unl = np.full(semi_l.shape[0],-1)
	label = np.concatenate((train_l,semi_unl),axis=0).astype('int')
	data = np.concatenate((train_d,semi_d),axis=0).astype('float')
	print('label size: ', label.shape)
	print('data size: ', data.shape)
	print('label propagation...')	
	#print(label)
	#model = LabelPropagation(kernel='knn',n_neighbors=5,max_iter=10000,tol=0.001,n_jobs=-1)
	model = LabelSpreading(kernel='rbf',gamma=20,alpha=0.2,n_neighbors=5,max_iter=100000,tol=0.001,n_jobs=20)
	model.fit(data,label)
	oursemi_l = model.predict(semi_d)

	ourlabel = np.concatenate((train_l,oursemi_l),axis=0)
	csvlabel = np.concatenate((train_l,semi_l),axis=0)
	print('our... ',ourlabel)
	print('csv... ',csvlabel)
	similarity = 0
	for i in range(len(ourlabel)):
		if ourlabel[i] == csvlabel[i]:
			similarity += 1
	print('new train num: ', similarity)
	print('ratio: ', similarity/len(ourlabel))
	return data ,name, ourlabel, csvlabel



def tsne(data, ourlabel, csvlabel, name):

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
	fig,(ax1,ax2)=plt.subplots(1,2)

	print('CSV genres...',len(dicCSV))
	colors = cm.rainbow(np.linspace(0, 1, int(len(dicCSV))))
	ax1.set_title('csv')
	for n,i in enumerate(dicCSV):
		temp = np.array(dicCSV[i])
		ax1.scatter(temp[:,0],temp[:,1],color=colors[n],s=5)
		#for k in range(temp.shape[0]):
			#axes[x,y].annotate(str(i),(temp[k,0],temp[k,1]))
			#axes[x,y].text(temp[k,0]*1.05,temp[k,1]*1.05,str(i),fontdict={'size': 7})


	print('OUR genres...',len(dicOUR))
	colors = cm.rainbow(np.linspace(0, 1, int(len(dicOUR))))
	ax2.set_title('our')
	for n,i in enumerate(dicOUR):
		temp = np.array(dicOUR[i])
		ax2.scatter(temp[:,0],temp[:,1],color=colors[n],s=5)
		#for k in range(temp.shape[0]):
			#axes[x,y].annotate(str(i),(temp[k,0],temp[k,1]))
			#axes[x,y].text(temp[k,0]*1.05,temp[k,1]*1.05,str(i),fontdict={'size': 7})

	fig.savefig('TSNE_cluster/three_genre/'+name+'.png')
	plt.close()
	

if __name__ == '__main__':
	warnings.filterwarnings("ignore")
	main()
