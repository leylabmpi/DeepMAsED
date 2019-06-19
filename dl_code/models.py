import numpy as np
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


class Generator(keras.utils.Sequence):
    def __init__(self, x, y, max_len, batch_size=32, shuffle=True): 
        'Initialization'
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_len = max_len
        self.x = x
        self.y = y
        self.shuffle = shuffle
        self.n_feat = x[0].shape[1]

        mean, std = utils.compute_mean_std(self.x)
        self.mean = mean
        self.std = std

        # Shuffle data
        self.indices = np.arange(len(x))
        if self.shuffle: 
            np.random.shuffle(self.indices)

        self.on_epoch_end()

    def on_epoch_end(self):
        """
        Reshuffle when epoch ends 
        """
        if self.shuffle: 
            np.random.shuffle(self.indices)


    def generate(self, indices_tmp):
        """
        Generate new mini-batch
        """

        x_mb = np.zeros((self.batch_size, self.max_len, self.n_feat, 1))
        y_mb = np.zeros((self.batch_size, 1))

        for i, idx in enumerate(indices_tmp):
            x_mb[i, 0:self.x[idx].shape[0]] = np.expand_dims(
              (self.x[idx] - self.mean) / self.std, -1)
            y_mb[i] = self.y[idx]

        return x_mb, y_mb

    def __len__(self):
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        """
        Get new mb
        """
        indices_tmp = \
          self.indices[self.batch_size * index : self.batch_size * (index + 1)]
        x_mb, y_mb = self.generate(indices_tmp)
        return x_mb, y_mb



            







