from keras.models import Sequential
from keras.models import Model 
from keras.layers import Dense, Activation, Flatten, Dropout, LeakyReLU, Input, GlobalMaxPooling2D
from keras.layers.advanced_activations import PReLU
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from para import *

def kaggleModel():
    inp = Input(shape=(fre_bin, max_time, 1))
    x = Conv2D(32, (4,10), padding="same")(inp)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Activation("selu")(x)
    x = MaxPooling2D()(x)

    x = Conv2D(32, (4,10), padding="same")(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Activation("selu")(x)
    x = MaxPooling2D()(x)

    x = Conv2D(32, (4,10), padding="same")(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Activation("selu")(x)
    x = MaxPooling2D()(x)

    x = Conv2D(32, (4,10), padding="same")(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Activation("selu")(x)
    x = MaxPooling2D()(x)

    x = Flatten()(x)
    x = Dense(64)(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Activation("selu")(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inp, outputs=out)
    return model

def vggModel_1():
    model = Sequential()
    model.add(Conv2D(16, (20, 8), padding='same', input_shape=(fre_bin, max_time, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.05))
    model.add(Conv2D(16, (20,8), padding='same', input_shape=(fre_bin, max_time, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.05))
    model.add(Conv2D(16, (20, 8), padding='same', input_shape=(fre_bin, max_time, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.05))
    model.add(MaxPooling2D(3,3))
    model.add(Dropout(0.2))

    model.add(Conv2D(32, (10,4), padding='same', input_shape=(fre_bin, max_time, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.05))
    model.add(Conv2D(32, (10,4), padding='same', input_shape=(fre_bin, max_time, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.05))
    model.add(Conv2D(32, (10,4), padding='same', input_shape=(fre_bin, max_time, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.05))
    model.add(MaxPooling2D(3,3))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (5,2), padding='same', input_shape=(fre_bin, max_time, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.05))
    model.add(Conv2D(64, (5,2), padding='same', input_shape=(fre_bin, max_time, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.05))
    model.add(Conv2D(64, (5,2), padding='same', input_shape=(fre_bin, max_time, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.05))
    model.add(MaxPooling2D(3,3))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3,1), padding='same', input_shape=(fre_bin, max_time, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.05))
    model.add(Conv2D(128, (3,1), padding='same', input_shape=(fre_bin, max_time, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.05))
    model.add(Conv2D(128, (3,1), padding='same', input_shape=(fre_bin, max_time, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.05))
    model.add(MaxPooling2D(3,3))
    model.add(Dropout(0.2))

    model.add(GlobalMaxPooling2D())

    model.add(Dense(4096, activation='selu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model

def vggModel_2():
    model = Sequential()
    model.add(Conv2D(32, (20, 8), padding='same', input_shape=(fre_bin, max_time, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.05))
    model.add(Conv2D(32, (20,8), padding='same', input_shape=(fre_bin, max_time, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.05))
    model.add(MaxPooling2D(3,3))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (10,4), padding='same', input_shape=(fre_bin, max_time, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.05))
    model.add(Conv2D(64, (10,4), padding='same', input_shape=(fre_bin, max_time, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.05))
    model.add(MaxPooling2D(3,3))
    model.add(Dropout(0.35))

    model.add(Conv2D(128, (5,2), padding='same', input_shape=(fre_bin, max_time, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.05))
    model.add(Conv2D(128, (5,2), padding='same', input_shape=(fre_bin, max_time, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.05))
    model.add(MaxPooling2D(3,3))
    model.add(Dropout(0.35))

    model.add(Conv2D(256, (3,1), padding='same', input_shape=(fre_bin, max_time, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.05))
    model.add(Conv2D(256, (3,1), padding='same', input_shape=(fre_bin, max_time, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.05))
    model.add(MaxPooling2D(3,3))
    model.add(Dropout(0.35))
    model.add(GlobalMaxPooling2D())

    model.add(Dense(1024, activation='selu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model

def myModel():
    model = Sequential()
    model.add(Conv2D(64, (20, 8), padding='same', input_shape=(fre_bin, max_time, 1))) #activation='selu'))
    #model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
    model.add(LeakyReLU(0.05))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.2))

    model.add(Conv2D(32, (10,4), padding='same', input_shape=(fre_bin, max_time, 1), activation='selu'))
    model.add(MaxPooling2D(3,3))
    model.add(Dropout(0.2))

    model.add(Conv2D(16, (5,2), padding='same', input_shape=(fre_bin, max_time, 1), activation='selu'))
    #model.add(LeakyReLU(0.05))
    model.add(MaxPooling2D(3,3))
    model.add(Dropout(0.2))


    model.add(Flatten())

    model.add(Dense(256, activation='selu'))
    model.add(Dropout(0.2))

    model.add(Dense(128, activation='selu'))
    model.add(Dropout(0.2))

    model.add(Dense(64, activation='selu'))
    model.add(Dropout(0.2))

    model.add(Dense(64, activation='selu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model
