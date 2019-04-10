import _pickle as pickle
import os 
import keras
from keras import backend as K
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

def class_recall(label):
    def metr(y_true, y_pred):
        #lass_id_true = K.argmax(y_true, axis=-1)
        #class_id_preds = K.argmax(y_pred, axis=-1)
        class_id_preds = K.cast(K.greater(y_pred, 0.5), 'int32')
        y_true = K.cast(y_true, 'int32')
        accuracy_mask = K.cast(K.equal(y_true, label), 'int32')
        class_acc_tensor = K.cast(K.equal(y_true, class_id_preds), 'int32') * accuracy_mask
        class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
        return class_acc
    return metr
