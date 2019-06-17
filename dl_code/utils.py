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

def compute_mean_std(x_tr):

    n_feat = x_tr[0].shape[1]
    feat_sum = np.zeros(n_feat)
    feat_sq_sum = np.zeros(n_feat)
    n_el = 0

    for xi in x_tr:
        sum_xi = np.sum(xi, 0)
        sum_sq = np.sum(xi ** 2, 0)
        feat_sum += sum_xi
        feat_sq_sum += sum_sq
        n_el += xi.shape[0]

    mean = feat_sum / n_el
    std = np.sqrt(feat_sq_sum / n_el - mean ** 2)

    return mean, std

def normalize(x, mean, std, max_len):
    """
    Given mean and std vector computed from training data, 
    normalize and return in shape (n, p, 1)
    """
    n_feat = x[0].shape[1]

    for i in range(len(x)):
        x[i] = (x[i] - mean) / std
        num_timesteps = x[i].shape[0]
        if num_timesteps < max_len:
            x[i] = np.concatenate((x[i], 
                                   np.zeros((max_len - num_timesteps, n_feat))), 
                                   0)
        else:
            x[i] = x[i][0:max_len]
        x[i] = np.expand_dims(x[i], 0)
    
    x = np.concatenate(x, 0)
    x = np.expand_dims(x, -1)
    return x


def load_features(data_path, max_len=3000, test_size=0.2,
                  standard=1, mode='chimera', technology='megahit'):
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
    dirs = os.listdir(data_path)
    for i, f in enumerate(dirs):#os.listdir(data_path):

        current_path = os.path.join(data_path, f, technology)

        if not os.path.exists(os.path.join(current_path, 'features.pkl')):
            print("Populating pickle file...")
            pickle_data(current_path, 'features.tsv.gz', 'features.pkl')

    x, y, ye, yext, n2i = [], [], [], [], []

    for i, f in enumerate(dirs):
        current_path = os.path.join(data_path, f, technology)
        with open(os.path.join(current_path, 'features.pkl'), 'rb') as f:
            xi, yi, yei, yexti, n2ii = pickle.load(f)

            if mode == 'extensive':
                yi = yexti

            # Remove two last features, always zero
            # Subsample data, eq number of 1 and 0
            yi = np.array(yi)
            where_one = np.where(yi == 1)[0]
            where_zero = np.where(yi == 0)[0]
            len_ones = len(where_one)

            for idx in where_one:
                x.append(xi[idx][:, 0:-2])

            for j, idx in enumerate(where_zero):
                if j == len_ones:
                    break
                x.append(xi[idx][:, 0:-2])

            yinew = np.array(len_ones * [1] + len_ones * [0])
            yext.append(yinew)

    #y = np.concatenate(y)
    #ye = np.concatenate(ye)
    yext = np.concatenate(yext)

    if mode == 'edit':
        y = 100 * np.array(ye)
    elif mode == 'extensive':
        y = yext

    #Split in train/test
    n_ex = len(x)
    idx = np.arange(n_ex)
    np.random.shuffle(idx)
    test_idx = idx[0:int(test_size * n_ex)]
    train_idx = idx[int(test_size * n_ex):]

    x_tr, y_tr, x_te, y_te = [], [], [], []
    for i in train_idx:
        x_tr.append(x[i])
        y_tr.append(y[i])
    for i in test_idx:
        x_te.append(x[i])
        y_te.append(y[i])

    y_tr = np.array(y_tr)
    y_te = np.array(y_te)

    #Compute normalization (mean and std)
    mean, std = compute_mean_std(x_tr)

    #Normalize
    x_tr = normalize(x_tr, mean, std, max_len)
    x_te = normalize(x_te, mean, std, max_len)

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

            feat.append(np.array([int(ri) for ri in row[4:w_chimera]])[None, :].astype(np.uint8))

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
