# import
## batteries
import logging
## 3rd party
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, BatchNormalization, AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2D, Flatten
## application
from DeepMAsED import Utils


class deepmased(object):
    """
    Implements a convolutional network for misassembly prediction. 
    """
    def __init__(self, config):
        self.max_len = config.max_len
        self.filters = config.filters
        self.n_conv = config.n_conv
        self.n_features = config.n_features
        self.pool_window = config.pool_window
        self.dropout = config.dropout
        self.lr_init = config.lr_init
        self.n_fc = config.n_fc
        self.n_hid = config.n_hid

        self.net = Sequential()

        self.net.add(Conv2D(self.filters, kernel_size=(2, self.n_features), 
                            input_shape=(self.max_len, self.n_features, 1),
                            activation='relu', padding='valid'))
        self.net.add(BatchNormalization(axis=-1))

        for i in range(1, self.n_conv):
            self.net.add(Conv2D(2 ** i * self.filters, kernel_size=(2, 1), 
                                strides=2, 
                                input_shape=(self.max_len, 1, 2 ** (i - 1) * self.filters), 
                                activation='relu'))
            self.net.add(BatchNormalization(axis=-1))
        
        self.net.add(AveragePooling2D((self.pool_window, 1)))
        self.net.add(Flatten())

        optimizer = keras.optimizers.Adam(lr=self.lr_init)

        # binary classification
        for _ in range(self.n_fc - 1):
            self.net.add(Dense(self.n_hid, activation='relu'))
            self.net.add(Dropout(rate=self.dropout))
            
        self.net.add(Dense(1, activation='sigmoid'))
        self.net.add(Dropout(rate=self.dropout))

        recall_0 = Utils.class_recall(0)
        recall_1 = Utils.class_recall(1)
        self.net.compile(loss='binary_crossentropy',
                         optimizer=optimizer,
                         metrics=[recall_0, recall_1])

        self.reduce_lr = keras.callbacks.ReduceLROnPlateau(
                               monitor='val_loss', factor=0.5,
                               patience=5, min_lr = 0.01 * self.lr_init)

    def predict(self, x):
        return self.net.predict(x)

    def predict_generator(self, x):
        return self.net.predict_generator(x)

    def print_summary(self):
        print(self.net.summary())

    def save(self, path):
        self.net.save(path)


class Generator(keras.utils.Sequence):
    def __init__(self, x, y, max_len=10000, batch_size=32,
                 shuffle=True, norm_raw=True,
                 mean_tr=None, std_tr=None): 
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_len = max_len
        self.x = x
        self.y = y
        self.shuffle = shuffle
        self.n_feat = x[0].shape[1]

        if mean_tr is None:
            mean, std = Utils.compute_mean_std(self.x)
            self.mean = mean
            self.std = std
            if not norm_raw:
                self.mean[0:4] = 0
                self.std[0:4] = 1
        else:
            self.mean = mean_tr
            self.std = std_tr

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

        x_mb = np.zeros((len(indices_tmp), self.max_len, self.n_feat, 1))
        y_mb = np.zeros((len(indices_tmp), 1))

        for i, idx in enumerate(indices_tmp):
            x_mb[i, 0:self.x[idx].shape[0]] = np.expand_dims(
              (self.x[idx] - self.mean) / self.std, -1)
            y_mb[i] = self.y[idx]

        return x_mb, y_mb

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        """
        Get new mb
        """
        if self.batch_size * (index + 1) < len(self.indices):
            indices_tmp = \
              self.indices[self.batch_size * index : self.batch_size * (index + 1)]
        else:
            indices_tmp = \
              self.indices[self.batch_size * index : ]  
        x_mb, y_mb = self.generate(indices_tmp)
        return x_mb, y_mb




