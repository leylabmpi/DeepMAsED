import keras
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, BatchNormalization, AveragePooling2D
from keras.layers import MaxPooling2D, Dropout
from keras.layers import Conv2D, Flatten

class chimera_net(object):
    def __init__(self, config):

        max_len = config.max_len
        filters = config.filters
        n_features = config.n_features
        pool_window = config.pool_window
        dropout = config.dropout

        self.net = Sequential()

        self.net.add(Conv2D(filters[0], kernel_size=(2, 9), 
                            input_shape=(max_len, n_features, 1), 
                            activation='relu', padding='same'))
        self.net.add(BatchNormalization(axis=-1))

        for i in range(1, len(filters)):
            self.net.add(Conv2D(filters[i], kernel_size=(2, 1), 
                                strides=2, 
                                input_shape=(max_len, 1, filters[i - 1]), 
                                activation='relu'))
            self.net.add(BatchNormalization(axis=-1))
        
        self.net.add(AveragePooling2D((pool_window, 1)))
        self.net.add(Flatten())
        self.net.add(Dense(1, activation='sigmoid'))
        self.net.add(Dropout(rate=dropout))

        optimizer = keras.optimizers.adam(lr=0.01)
        self.net.compile(loss='binary_crossentropy', optimizer=optimizer)

    def predict(self, x):
        return self.net.predict(x)

    def print_summary(self):
        print(self.net.summary())
