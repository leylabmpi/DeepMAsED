import _pickle as pickle
import os 
import keras
from keras import backend as K
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import csv
import gzip
import IPython


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

    # Pre-process once if not done already
    if not os.path.exists(os.path.join(data_path, 'features.pkl')):
        print("Populating pickle file...")
        pickle_data(data_path, 'features.tsv.gz', 'features.pkl')

    with open(os.path.join(data_path, 'features.pkl'), 'rb') as f:
        x, y, ye, yext, n2i = pickle.load(f)

    if mode == 'edit':
        y = 100 * np.array(ye)
    elif mode == 'extensive':
        y = yext

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

def pickle_data(data_path, features_in, features_out):  
    """
    One time function parsing the csv file and dumping the 
    values of interest into a pickle file. 
    """
    feat_contig, target_contig, target_contig_edit = [], [], []
    target_contig_ext = []
    name_to_id = {}

    idx = 0
    #Read tsv and process features
    with gzip.open(os.path.join(data_path, features_in), 'rt') as f:

        tsv = csv.reader(f, delimiter='\t')
        col_names = next(tsv)

        w_chimera = col_names.index('chimeric')
        w_edit = col_names.index('edit_dist_norm')
        w_ext = col_names.index('Extensive_misassembly')

        prev_name, tgt, tgt_ed = None, None, None
        feat = []
        check_size = set([])

        for row in tsv:

            check_size.add(row[0])

            if prev_name is None: 
                prev_name = row[0]
            if tgt is None: 
                tgt = row[w_chimera]
                tgt_ed = row[w_edit]
                tgt_ext = row[w_ext]

            if row[0] != prev_name:

                prev_name = row[0]
                if tgt == '':
                    tgt = None
                    tgt_edit = None
                    tgt_ext = None
                    feat = []
                    continue

                feat_contig.append(np.concatenate(feat, 0))

                if tgt == 'FALSE':
                    target_contig.append(0)
                else:
                    target_contig.append(1)

                target_contig_edit.append(float(tgt_ed))

                if tgt_ext == '':
                    target_contig_ext.append(0)
                else:
                    target_contig_ext.append(1)

                feat = []
                tgt = None
                tgt_ext = None

            feat.append(np.array([int(ri) for ri in row[4:w_chimera]])[None, :])

            if row[0] not in name_to_id:
                name_to_id[row[0]] = idx
                idx += 1

    # Save processed data into pickle file
    print("Total number of contigs: %d" % (len(check_size) - 2))
    print("Number of processed labels: %d" % len(target_contig_ext))

    with open(os.path.join(data_path, features_out), 'wb') as f:
        pickle.dump([feat_contig, target_contig, target_contig_edit, target_contig_ext, name_to_id], f)

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
