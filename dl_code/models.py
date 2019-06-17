import keras
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, BatchNormalization, AveragePooling2D
from keras.layers import MaxPooling2D, Dropout
from keras.layers import Conv2D, Flatten
import utils

class chimera_net(object):
    """
    Implements a convolutional network for chimera prediction. 
    """

    def __init__(self, config):

        max_len = config.max_len
        filters = config.filters
        n_conv = config.n_conv
        n_features = config.n_features
        pool_window = config.pool_window
        dropout = config.dropout
        lr_init = config.lr_init
        mode = config.mode
        n_fc = config.n_fc
        n_hid = config.n_hid

        self.net = Sequential()

        self.net.add(Conv2D(filters, kernel_size=(2, 7), 
                            input_shape=(max_len, n_features, 1), 
                            activation='relu', padding='same'))
        self.net.add(BatchNormalization(axis=-1))

        for i in range(1, n_conv):
            self.net.add(Conv2D(2 ** i * filters, kernel_size=(2, 1), 
                                strides=2, 
                                input_shape=(max_len, 1, 2 ** (i - 1) * filters), 
                                activation='relu'))
            self.net.add(BatchNormalization(axis=-1))
        
        self.net.add(AveragePooling2D((pool_window, 1)))
        self.net.add(Flatten())

        optimizer = keras.optimizers.adam(lr=lr_init)

        if mode in ['chimera', 'extensive']:
            for _ in range(n_fc - 1):
                self.net.add(Dense(n_hid, activation='relu'))
                self.net.add(Dropout(rate=dropout))

            self.net.add(Dense(1, activation='sigmoid'))
            self.net.add(Dropout(rate=dropout))

            recall_0 = utils.class_recall(0)
            recall_1 = utils.class_recall(1)
            self.net.compile(loss='binary_crossentropy', optimizer=optimizer,
                             metrics=[recall_0, recall_1])
        elif mode == 'edit':
            self.net.add(Dense(20, activation='relu'))
            self.net.add(Dropout(rate=dropout))
            self.net.add(Dense(20, activation='relu'))
            self.net.add(Dropout(rate=dropout))
            self.net.add(Dense(1, activation='linear'))
            self.net.compile(loss='mean_absolute_error', optimizer=optimizer,
                             metrics=[utils.explained_var])
        else:
            raise('Training mode not supported.')

        self.reduce_lr = keras.callbacks.ReduceLROnPlateau(
          monitor='val_loss', factor=0.5, patience=5, min_lr = 0.01 * lr_init)

    def predict(self, x):
        return self.net.predict(x)

    def print_summary(self):
        print(self.net.summary())

    def save(self, path):
        self.net.save(path)
