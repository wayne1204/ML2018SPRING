import sys
import numpy as np
from utils import *
from keras.models import load_model
from para import *
import argparse
import keras.backend.tensorflow_backend as K
import tensorflow as tf
import pandas as pd


dm = DataManager()
dm.add_rawData('data/sample_submission.csv', mode='test')
dm.preprocess()
#dm.dump_data(test_spectrum_mic_ex_dir)
X = np.array(([d.reshape(fre_bin, max_time, 1) for d in dm.get_data('test_data')[0]])) / -80.
length = len(X)
print(X.shape)


def test(model):
    global X
    model = load_model(model)
    model.summary()
    X_1 = np.array(X[:int(length//4)])
    y_prob_1 = model.predict(X_1, batch_size=32, verbose=True)
    X_1 = None

    X_1 = np.array(X[int(length//4):int(length*2//4)])
    y_prob_2 = model.predict(X_1, batch_size=32, verbose=True)
    X_1 = None

    X_1 = np.array(X[int(2*length//4):int(3*length//4)])
    y_prob_3 = model.predict(X_1, batch_size=32, verbose=True)
    X_1 = None

    X_1 = np.array(X[int(3*length//4):])
    y_prob_4 = model.predict(X_1, batch_size=32, verbose=True)
    X_1 = None

    y_prob = np.concatenate((y_prob_1, y_prob_2, y_prob_3, y_prob_4), axis=0)
    return y_prob
parser = argparse.ArgumentParser(description='Sound classification')
parser.add_argument('--result_path', type=str)
parser.add_argument('--gpu_fraction', default=0.3, type=float)
parser.add_argument('--models', type=str)
args = parser.parse_args()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)
K.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

models = []
num_model = len(args.models.split(','))
models = (args.models.split(','))
predc = 0
for i in range(num_model):
    prob = test(models[i])
    np.save('model/'+str(models[i])+'.npy',prob)
    predc  += prob
    #models.append(sys.argv[i+2])
#predc = predc / float(num_model)
#ensembleModels(models)
#top_3 = np.array(category)[np.argsort(-predc, axis=1)[:, :3]]
#predicted_labels = [' '.join(list(x)) for x in top_3]
#prd_class = y_prob.argmax(axis=-1)
#print(prd_class)
'''
data = pd.read_csv('../data/sample_submission.csv').values

for i, ele in enumerate(predicted_labels):
    data[i][1] = ele

columns = ['fname', 'label']
df = pd.DataFrame(data=data, columns=columns)
df.to_csv(args.result_path, encoding='utf-8', index=False)
'''

