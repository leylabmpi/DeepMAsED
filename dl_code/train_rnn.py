import numpy as np
import keras
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, BatchNormalization, AveragePooling2D
from keras.layers import Conv2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import os
import _pickle as pickle

#hidden_dim = 5
#
#inputs = Input(shape=(None, 5))
#encoder = LSTM(hidden_dim, return_state=True)
#enc_out, h, c = encoder(inputs)
#
#output = Dense(units = 2, activation='relu', input_dim=hidden_dim)(h)
#
#model = Model(inputs, output)
#
#model.compile(loss='categorical_crossentropy', optimizer='adam')

model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 9), input_shape=(3000, 9, 1), 
          activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(AveragePooling2D((50, 1)))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

print(model.summary())

data_path='/home/mrojas/deepmased/tests/output_n10/map/1/megahit'
with open(os.path.join(data_path, 'features.pkl'), 'rb') as f:
    x, y, n2i = pickle.load(f)

x = keras.preprocessing.sequence.pad_sequences(x, maxlen=3000)
x = np.expand_dims(x, -1)
y = np.array(y)

x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2)
model.fit(x_tr, y_tr, validation_data=(x_te, y_te), epochs=5)

pred_tr = (model.predict(x_tr) > 0.5).astype(int)
pred_te = (model.predict(x_te) > 0.5).astype(int)

print("Training")
print(confusion_matrix(pred_tr, y_tr))
print("Test")
print(confusion_matrix(pred_te, y_te))

import IPython
IPython.embed()
