import _pickle as pickle
import os 
import keras
import numpy as np
from sklearn.model_selection import train_test_split

def load_features(data_path, max_len=3000, test_size=0.2):

    with open(os.path.join(data_path, 'features.pkl'), 'rb') as f:
        x, y, n2i = pickle.load(f)

    x = keras.preprocessing.sequence.pad_sequences(x, maxlen=max_len)
    x = np.expand_dims(x, -1)
    y = np.array(y)

    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=test_size, 
                                              random_state=1)

    return x_tr, x_te, y_tr, y_te
