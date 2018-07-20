import numpy as np
import pandas as pd
from para import *
import librosa 
from keras.utils import to_categorical
import random
from os import listdir
import scipy

class DataManager:
    def __init__(self):
        self.data = {}

    # Load raw data name from csv file and load the wav by librosa
    def add_rawData(self, data_path, mode='train'):
        print ('  Loading data from %s...'%data_path)
        data = pd.read_csv(data_path).values
        train_x, train_y, train_fname = [], [], []
        semi_x, semi_y, semi_fname = [], [], []
        test, test_fname = [], []
        count = 0
        for line in data:
            if(mode =='train' or mode == 'semi'):
                x, sr = np.squeeze(librosa.load(train_dir+line[0], sr=44100))
            if(mode =='test'):
                x, sr = np.squeeze(librosa.load(test_dir+line[0], sr=44100))
            print('    loaded num: {}\r'.format(count), end='')
            #if count == 100: break
            if mode == 'train':
                if(line[2]):
                    train_x.append(x)
                    train_y.append(category.index(line[1]))
                    train_fname.append(line[0])
                    count += 1
            elif mode == 'semi': 
                if(line[2]):
                    train_x.append(x)
                    train_y.append(category.index(line[1]))
                    train_fname.append(line[0])
                else:
                    semi_x.append(x)
                    semi_y.append(category.index(line[1]))
                    semi_fname.append(line[0])
                count += 1
            else:
                test.append(x)
                test_fname.append(line[0])
                count += 1
        if mode == 'train':
            self.data['train_data'] = np.array([train_x, train_y, train_fname])
            #print(self.data['train_data'][0].shape)
        elif mode == 'semi':
            self.data['train_data'] = np.array([train_x, train_y, train_fname])
            self.data['semi_data'] = np.array([semi_x, semi_y, semi_fname])
        else:
            self.data['test_data'] = np.array([test, test_fname])
        print(" ")

    def get_data(self, name):
        return self.data[name]

    def load_data(self, path, mode='train'):
        print ('  Loading data from %s...'%path)
        f = listdir(path)
        train_x, train_y, train_fname = [], [], []
        semi_x, semi_y, semi_fname = [], [], []
        test, test_fname = [], []
        count = 0
        for idx, fname in enumerate(f):
            tmp = np.load(path+fname)
            tmp_fname = str(fname.split('.')[0])+'.wav'
            #print(tmp_fname)
            print('    loaded num: {}\r'.format(count), end='')
            #if count == 6000: break
            if(mode == 'train'):
                if(tmp[2]): 
                    train_x.append(tmp[0])
                    train_y.append(tmp[1])
                    train_fname.append(tmp_fname)
                    count += 1
            elif(mode == 'semi'):
                if(tmp[2]): 
                    train_x.append(tmp[0])
                    train_y.append(tmp[1])
                    train_fname.append(tmp_fname)
                    count += 1
                else:
                    semi_x.append(tmp[0])
                    semi_y.append(tmp[1])
                    semi_fname.append(tmp_fname)
                    count += 1
            else:
                test.append(tmp[0])
                test_fname.append(tmp_fname)
                count += 1
        if mode == 'train':
            self.data['train_data'] = np.array([train_x, train_y, train_fname])
        elif mode == 'semi':
            self.data['train_data'] = np.array([train_x, train_y, train_fname])
            self.data['semi_data'] = np.array([semi_x, semi_y, semi_fname])
        else:
            self.data['test_data'] = np.array([test, test_fname])
        print(" ")
            
    def dump_data(self, path, mode='train'):
        print ('  Dumping data to %s...'%path)
        if(mode == 'train') :
            for key in self.data:
                if(key == 'train_data'):
                    for image, label, fname in zip(self.data[key][0], self.data[key][1], self.data[key][2]):
                        np.save('{}{}.npy'.format(path, fname), [image, label, 1])
                else:
                    for image, label, fname in zip(self.data[key][0], self.data[key][1], self.data[key][2]):
                        np.save('{}{}.npy'.format(path, fname), [image, label, 0])
        else:
            for key in self.data:
                for image, fname in zip(self.data[key][0], self.data[key][1]):
                    np.save('{}{}.npy'.format(path, fname), [image])
    # Preprocess for all data and save to train_image dir
    #  you can edit you preprocess method here
    #  train_data, test_data, semi_data
    def preprocess(self, mode='spectrum', load=False):
        print('  Preprocessin data in {} mode'.format(mode))
        #print(self.data['test_data'][0].shape)
        count = 0
        for key in self.data:
            for idx, sig in enumerate(self.data[key][0]):
                print('    Preprocessed num: {}\r'.format(count), end='')
                if(len(sig) == 0):
                    self.data[key][0][idx] = np.full((fre_bin, 1), -80)
                else:
                    D = librosa.stft(sig, n_fft=n_fft, hop_length=hop_length) 
                    self.data[key][0][idx] = librosa.amplitude_to_db(librosa.magphase(D)[0], ref=np.max)
                count += 1
        print(" ")
        sum_frame= 0
        count = 0
        for key in self.data:
            for idx, sig in enumerate(self.data[key][0]):
                sum_frame += len(sig[0])
                count += 1
        avg_frame = sum_frame // count
        print('    Average time frame : {}'.format(avg_frame))
        for key in self.data:
            for idx, sig in enumerate(self.data[key][0]):
                self.data[key][0][idx] = scipy.misc.imresize((sig*255/(-80.0)), (fre_bin, max_time), interp='bicubic') / 255 * (-80.0)
    def split_data(self, data, ratio):
        X = np.array([d.reshape(fre_bin, max_time, 1) for d in data[0]])
        #print(X.shape)
        Y = to_categorical(data[1], num_classes)
        pairs = list(zip(X, Y))
        random.shuffle(pairs)
        X, Y = zip(*pairs)
        X, Y = np.array(X), np.array(Y)
        data_size = len(X)
        val_size = int(data_size * ratio)
        return (X[val_size:],Y[val_size:]),(X[:val_size],Y[:val_size])

    def get_semi_data(self,data_name,label,labelprob,threshold,loss_function): 
        # if th==0.3, will pick label>0.7 and label<0.3
        labelprob = np.squeeze(labelprob)
        print('semi size: ',labelprob.shape)
        semi_X = []
        semi_Y = []
        for n, data in enumerate(labelprob):
            if data > 1-threshold and data < 0.9:
                semi_X.append(data_name[n])
                semi_Y.append(label[n])

        #index = (labelprob>(1-threshold)) and (labelprob < 0.9)
        if loss_function=='binary_crossentropy':
            print(semi_X[index].shape)
            return semi_X[index], semi_Y[index]
        elif loss_function=='categorical_crossentropy':
            return semi_X, semi_Y
        else :
            raise Exception('Unknown loss function : %s'%loss_function) 
