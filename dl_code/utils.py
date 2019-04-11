import _pickle as pickle
import os 
import keras
from keras import backend as K
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_features(data_path, max_len=3000, test_size=0.2, mode='chimera'):
    """
    Loads features, pre-process them and returns training and test data. 

    Inputs: 
        data_path: path to directory containing features.pkl
        max_len: fixed length of contigs
        test_size: portion of the data kept for testing

    Outputs:
        x_tr, x_te : train and test features
        y_tr, y_te: train and test targets. 
    """

    with open(os.path.join(data_path, 'features.pkl'), 'rb') as f:
        x, y, ye, n2i = pickle.load(f)

    if mode == 'edit':
        y = 100 * np.array(ye)

    x = keras.preprocessing.sequence.pad_sequences(x, maxlen=max_len)
    y = np.array(y)

    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=test_size, 
                                              random_state=1)

    std_sc = StandardScaler()
    x_tr_shape = x_tr.shape
    x_tr = x_tr.reshape((x_tr_shape[0] * x_tr_shape[1], x_tr_shape[-1]))
    x_tr = std_sc.fit_transform(x_tr)
    x_tr = x_tr.reshape(x_tr_shape)
    x_tr = np.expand_dims(x_tr, -1)

    x_te_shape = x_te.shape
    x_te = x_te.reshape((x_te_shape[0] * x_te_shape[1], x_te_shape[-1]))
    x_te = std_sc.transform(x_te)
    x_te = x_te.reshape(x_te_shape)
    x_te = np.expand_dims(x_te, -1)

    y_tr = y_tr[:, None]
    y_te = y_te[:, None]

    return x_tr, x_te, y_tr, y_te

def class_recall(label):
    """
    Custom metric for Keras, computes recall per class. 

    Inputs:
        label: label wrt which recall is to be computed. 
    """
    def metr(y_true, y_pred):
        class_id_preds = K.cast(K.greater(y_pred, 0.5), 'int32')
        y_true = K.cast(y_true, 'int32')
        accuracy_mask = K.cast(K.equal(y_true, label), 'int32')
        class_acc_tensor = K.cast(K.equal(y_true, class_id_preds), 'int32') * accuracy_mask
        class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
        return class_acc
    return metr

def explained_var(y_true, y_pred):
    """
    Custom metric for Keras, explained variance.  

    """
    return 1  - K.mean((y_true - y_pred) ** 2) / K.var(y_true)
