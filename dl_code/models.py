import keras
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, BatchNormalization, AveragePooling2D
from keras.layers import MaxPooling2D
from keras.layers import Conv2D, Flatten

class chimera_net(object):
    def __init__(self, max_len):
        self.net = Sequential()
        self.net.add(Conv2D(16, kernel_size=(2, 9), input_shape=(max_len, 9, 1), 
                             activation='relu', padding='same'))
        self.net.add(BatchNormalization(axis=-1))
        self.net.add(Conv2D(32, kernel_size=(2, 1), strides=2, 
                              input_shape=(max_len, 1, 16), 
                              activation='relu'))
        self.net.add(BatchNormalization(axis=-1))
        self.net.add(AveragePooling2D((5, 1)))
        self.net.add(Flatten())
        self.net.add(Dense(1, activation='sigmoid'))

        optimizer = keras.optimizers.adam()
        self.net.compile(loss='binary_crossentropy', optimizer=optimizer)
