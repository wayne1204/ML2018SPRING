# FileName     [ visualizeFilter.py ]
# Date		   [ 2018.4 ]
# Synopsis     [ visualize the output of filter in convolution layer]

import csv
import os
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
from utils import *
# from marcos import *
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

def main(): 
    parser = argparse.ArgumentParser(prog='visualize_filter.py',
            description='ML-Assignment3 filter visualization.')
    parser.add_argument('--model', type=str, metavar='<#model>', required=True)
    parser.add_argument('--data', type=str, metavar='<#data>', required=True)
    # parser.add_argument('--attr', type=str, metavar='<#attr>', required=True)
    parser.add_argument('--filter_dir', type=str, metavar='<#filter_dir>', default='./image/filter')
    args = parser.parse_args()

    data_name = args.data
    # attr_name = args.attr
    model_name = args.model
    filter_dir = args.filter_dir

    print('load model')
    emotion_classifier = load_model(model_name)
    layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers)

    print('load data') 
    X = parsingData(data_name)

    # print('load attr')
    # attr = np.load(attr_name)
    # mean, std = attr[0], attr[1]

    input_img = emotion_classifier.input
    name_ls = ['conv2d_1','conv2d_3','conv2d_5']
    collect_layers = [ K.function([input_img, K.learning_phase()], [layer_dict[name].output]) for name in name_ls ]

    choose_id = -4997
    for cnt, fn in enumerate(collect_layers):
        photo = X[choose_id].reshape(1, 48, 48, 1)
        # im = fn;
        im = fn([photo, 0]) #get the output of that layer
        fig = plt.figure(figsize=(14, 8))
        #nb_filter = im[0].shape[3]
        nb_filter = 32
        for i in range(nb_filter):
            ax = fig.add_subplot(nb_filter/8, 8, i+1)
            ax.imshow(im[0][0, :, :, i], cmap='YlOrBr')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.xlabel('filter {}'.format(i))
            plt.tight_layout()
        # fig.suptitle('Output of layer{} (Given image{})'.format(cnt, choose_id))
        img_path = os.path.join(filter_dir, 'vis')
        if not os.path.isdir(img_path):
            os.mkdir(img_path)
        fig.savefig(os.path.join(img_path, '{}-{}'.format(name_ls[cnt], choose_id)))

if __name__ == '__main__':
    main()
