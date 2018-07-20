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
        #if count == 1000: break
        if mode == 'train':
            if(line[2]):
                train_y_fname[line[0]] = category.index(line[1])
                count += 1
        elif mode == 'semi':
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
        data['train_data'] = train_y_fname
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
        val_key.append(random.choice(list(data.items()))[0])
    for key in data:
        if(key not in val_key):
            train_key.append(key)
    partition['train'] = train_key
    partition['validation'] = val_key
    return partition



args = parse_args()
K.set_session(get_session(args.gpu_fraction))

'=========================='
'=== Data preprocessing ==='
'=========================='
'''
dm = DataManager()
if args.load:
    dm.load_data(train_spectrum_dir, mode='semi')
    (X, Y), (X_val, Y_val) = dm.split_data(dm.get_data('train_data'), 0.1)
    (X_semi, Y_semi), (X_val_semi, Y_val_semi) = dm.split_data(dm.get_data('semi_data'), 0.1)
else:
    dm.add_rawData('../data/train.csv', mode='semi')
    dm.preprocess()
    (X, Y), (X_val, Y_val) = dm.split_data(dm.get_data('train_data'), 0.1)
    (X_semi, Y_semi), (X_val_semi, Y_val_semi) = dm.split_data(dm.get_data('semi_data'), 0.1)
    #print((X.shape))


X = np.concatenate((X, X_semi), axis=0)
Y = np.concatenate((Y, Y_semi), axis=0)

X_val = np.concatenate((X_val, X_val_semi), axis=0)
Y_val = np.concatenate((Y_val, Y_val_semi), axis=0)
X = X / -80.0
X_val = X_val / -80.0

print("  Train data shape  = ", X.shape)
print("  Validation data shape = ", X_val.shape)
'''

train_set = read_csv(args.train_csv, mode='train')['train_data']
partition = split_data(train_set)
labels = train_set

training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)



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

adam = Adam()
print ('compile model...')
model.compile(adam, 'categorical_crossentropy', metrics=['accuracy'])
save_path = save_dir + "weights.{epoch:03d}-{val_acc:.5f}.h5"

csvlogger = CSVLogger('log/{}.csv'.format(args.model))
earlystopping = EarlyStopping(monitor='val_acc', patience = 15, verbose=1, mode='max')
checkpoint = ModelCheckpoint(filepath=save_path, verbose=1, save_best_only=True,
                             monitor='val_acc', mode='max' )
#model = load_model('../ckpt/new_vgg_train_music/weights.047-0.83456.h5')
#history = model.fit(X, Y, validation_data=(X_val, Y_val), epochs=100, batch_size=16, callbacks=[checkpoint, earlystopping, csvlogger] )

print("train on {} samples, validate on {} samples...".format(len(partition['train']),
                                                              len(partition['validation'])))

model.fit_generator(generator=training_generator,
                    epochs=epochs, steps_per_epoch=1*len(labels)//batch_size,
                    validation_data=validation_generator,
                    callbacks=[checkpoint, earlystopping, csvlogger],
                    validation_steps=1*len(partition['validation'])//batch_size,
                    use_multiprocessing=True,
                    workers=12)
