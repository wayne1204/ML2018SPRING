from utils import *
from dataGen import *
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.optimizers import Adam
import keras.backend.tensorflow_backend as K
import tensorflow as tf
import argparse 
from keras.preprocessing.image import ImageDataGenerator
from model import *

def parse_args():
    parser = argparse.ArgumentParser(description='Sound classification')
    parser.add_argument('model')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--train_csv', default='data/newtrain.csv', type=str)
    parser.add_argument('--gpu_fraction', default=1, type=float)
    return parser.parse_args()

def get_session(gpu_fraction=0.8):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def read_csv(data_path, mode='train'):
    print ('  Loading data from %s...'%data_path)
    data_csv = pd.read_csv(data_path).values
    train_y_fname = {}
    test_fname = []
    count = 0
    for line in data_csv:
        print('    loaded num: {}\r'.format(count), end='')
        #if count == 500: break
        if mode == 'train':
            if(line[2]):
                train_y_fname[line[0]] = category.index(line[1])
                count += 1
        elif mode == 'semi':
            if(line[2]==0):
                train_y_fname[line[0]] = category.index(line[1])
                count += 1
        else:
            test_fname.append(line[0])
            count += 1
    data = {}
    if mode == 'train':
        data['train_data'] = train_y_fname
        #print(self.data['train_data'][0].shape)
    elif mode == 'semi':
        data['semi_data'] = train_y_fname
    else:
        data['test_data'] = np.array([test_fname])
    print(" ")
    return data

def split_data(data, fraction=0.1):
    length = len(data)
    points = int(length*fraction)
    # get random key for # of points
    val_key = []
    partition = {}
    train_key = []
    for i in range(points):
        f = (random.choice(list(data.items()))[0])
        while f in val_key:
            f = (random.choice(list(data.items()))[0])
        val_key.append(f)

    for key in data:
        if(key not in val_key):
            train_key.append(key)
    partition['train'] = train_key
    partition['validation'] = val_key
    return partition


def split_pred(X, model):
    length = len(X)
    #print(np.array(X).shape)
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

    top_label = np.squeeze(np.argsort(-y_prob, axis=1)[:, :1])
    top_prob = np.squeeze(np.sort(-y_prob, axis=1)[:, :1]*(-1))
    print(top_prob[:10])
    return top_label, top_prob


args = parse_args()
K.set_session(get_session(args.gpu_fraction))

'=========================='
'=== Data preprocessing ==='
'=========================='

train_set = read_csv(args.train_csv, mode='train')['train_data']
trainvalkey = split_data(train_set)
#training_generator = DataGenerator(trainvalkey['train'], train_set, **params)
validation_generator = DataGenerator(trainvalkey['validation'], train_set, **params)


dm = DataManager()
dm.add_rawData(args.train_csv, mode='semi')
dm.preprocess()
semi_data = np.array(([d.reshape(fre_bin, max_time, 1) for d in dm.get_data('semi_data')[0]])) / -80.
#dm.dump_data(train_spectrum_mic_ex_dir)

#dm.load_data(train_spectrum_mic_ex_dir ,mode='semi')
#semi_data = []
#temp_data = dm.get_data('semi_data')[0] 
#for i in range(len(temp_data)):
#    semi_data.append(temp_data[i])
#semi_data = np.expand_dims(np.array(semi_data), axis=3) / -80.
print(semi_data.shape)    
semi_name = dm.get_data('semi_data')[2]


save_dir = '{}{}/'.format(save_dir ,args.model)
print(save_dir)
input()
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

'======================'
'=== Model building ==='
'======================'

model = vggModel_2()
model.summary()
print ('compile model...')
model.compile(Adam(), 'categorical_crossentropy', metrics=['accuracy'])
#model = load_model('../ckpt/new_vgg_train_music/weights.047-0.83456.h5')
#history = model.fit(X, Y, validation_data=(X_val, Y_val), epochs=100, batch_size=16, callbacks=[checkpoint, earlystopping, csvlogger] )

print("train on {} samples, validate on {} samples, semi on {} samples...".format(len(trainvalkey['train']),
                                                                                  len(trainvalkey['validation']),
                                                                                  len(semi_data)))
for i in range(30):

    threshold = (0.65 - i * 0.01)
    semi_pred_label, semi_pred_prob = split_pred(semi_data, model)
    semi_X, semi_Y =  dm.get_semi_data(semi_name,semi_pred_label,semi_pred_prob,threshold,'categorical_crossentropy')

    all_set = {}
    value1 = []
    value2 = []
    for key in trainvalkey['train']:
        all_set[key]=train_set[key]
        value1.append(train_set[key])

    for j,key in enumerate(semi_X):
        all_set[key]=semi_Y[j]
        value2.append(semi_Y[j])

    print('=========================')
    print(np.array(value1).shape)
    print(value1[:10])
    print(np.array(value2).shape)
    print(value2[:10])
    print ('-- iteration %d  all_data size: %d' % (i,len(all_set))) 
    print('=========================')
    
    semi_generator = DataGenerator(list(all_set.keys()), all_set, **params)

    save_path = save_dir+str(i)+'_'+str(len(all_set))+"-weights.{epoch:03d}-{val_acc:.5f}.h5"
    csvlogger = CSVLogger('log/semi/'+str(i)+'_{}.csv'.format(args.model))
    earlystopping = EarlyStopping(monitor='val_acc', patience = 15, verbose=1, mode='max')
    checkpoint = ModelCheckpoint(filepath=save_path, verbose=1, save_best_only=True,
                                 monitor='val_acc', mode='max' )

    model.fit_generator(generator=semi_generator,
                        epochs=20, steps_per_epoch=1*len(all_set)//batch_size,
                        validation_data=validation_generator,
                        callbacks=[checkpoint, earlystopping, csvlogger],
                        validation_steps=1*len(trainvalkey['validation'])//batch_size,
                        use_multiprocessing=True,
                        workers=12)
    '''
    folders_subjects = os.listdir(save_dir)
    folders_subjects.sort(reverse=True)
    filename = save_dir+str(folders_subjects[0])
    print ('load model from %s' % filename)
    model = load_model(filename)    
    '''