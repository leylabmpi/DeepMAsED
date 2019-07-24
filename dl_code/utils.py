import _pickle as pickle
import os 
import keras
from keras import backend as K
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import csv
from collections import defaultdict
import gzip
import IPython

def compute_mean_std(x_tr):
    """
    Given training data (list of contigs), compute mean and std 
    feature-wise. 
    """

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
    DETERIORATED
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

def load_features(data_path, max_len=10000, 
                  standard=1, mode='chimera', technology='megahit', 
                  pickle_only=False):
    """
    Loads features, pre-process them and returns training and test data. 

    Inputs: 
        data_path: path to directory containing features.pkl
        max_len: fixed length of contigs

    Outputs:
        x, y, i2n: lists, where each element comes from one metagenome, and 
          a dictionary with idx -> (metagenome, contig_name)
    """

    # Pre-process once if not done already
    dirs = os.listdir(data_path)
    for i, f in enumerate(dirs):#os.listdir(data_path):
        current_path = os.path.join(data_path, f, technology)

        if not os.path.exists(os.path.join(current_path, 'features.pkl')):
        #if True:
            print("Populating pickle file...")
            pickle_data_b(current_path, 'features.tsv.gz', 'features_new.pkl')

    if pickle_only: 
        exit()

    x, y, ye, yext, n2i = [], [], [], [], []
    shift = 0
    i2n_all = {}
    for i, f in enumerate(dirs):
        current_path = os.path.join(data_path, f, technology)
        with open(os.path.join(current_path, 'features_new.pkl'), 'rb') as feat:
            #xi, yi, yei, yexti, n2ii = pickle.load(feat)
            xi, yi, n2ii = pickle.load(feat)
            i2ni = reverse_dict(n2ii)

            #if mode == 'extensive':
            #    yi = yexti
            
            x_in_contig, y_in_contig = [], []

            n2i_keys = set([])
            for j in range(len(xi)):
                len_contig = xi[j].shape[0]
                #xi[j] = xi[j][0:max_len, 1:]

                idx_chunk = 0
                while idx_chunk * max_len < len_contig:
                    chunked = xi[j][idx_chunk * max_len : (idx_chunk + 1) * max_len, 1:]
            
                    x_in_contig.append(chunked)
                    y_in_contig.append(yi[j])

                    i2n_all[len(x_in_contig) - 1 + shift] = (int(f), i2ni[j][0])
                    idx_chunk += 1
                    n2i_keys.add(i2ni[j][0])

            # Each element is a metagenome
            x.append(x_in_contig)
            yext.append(np.array(y_in_contig))

            #Sanity check
            assert(len(n2i_keys - set(n2ii.keys())) == 0)
            assert(len(set(n2ii.keys()) - n2i_keys) == 0)

            shift = len(i2n_all)


    #y = np.concatenate(y)
    #ye = np.concatenate(ye)
    #yext = np.concatenate(yext)

    if mode == 'edit':
        y = 100 * np.array(ye)
    elif mode == 'extensive':
        y = yext
    else:
        raise("Mode currently not supported")

    return x, y, i2n_all

def kfold(x, y, idx_lo, k=5):

    # Define validation data
    x_tr, y_tr = [], []
    x_val, y_val = [], []

    meta_per_fold = int(len(x) / k)
    lower = idx_lo * meta_per_fold
    upper = (idx_lo + 1) * meta_per_fold

    for i, xi in enumerate(x):
        if i < lower or i >= upper: # idx_lo:
            x_tr = x_tr + xi
            y_tr.append(y[i])
        else:
            x_val = x_val + xi
            y_val.append(y[i])

    y_tr = np.concatenate(y_tr)
    y_val = np.concatenate(y_val)

    return x_tr, x_val, y_tr, y_val

def leave_one_out(x, y, idx_lo):

    # Define validation data
    x_val, y_val = x[idx_lo], y[idx_lo]
    x_tr, y_tr = [], []

    for i, xi in enumerate(x):
        if i == idx_lo:
            continue
        x_tr = x_tr + xi
        y_tr.append(y[i])

    y_tr = np.concatenate(y_tr)

    return x_tr, x_val, y_tr, y_val

    #Split in train/test
    """
    n_ex = len(x_tr)
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
    """
    #Compute normalization (mean and std)
    mean, std = compute_mean_std(x_tr)

    #Normalize
    x_tr = normalize(x_tr, mean, std, max_len)[:, :,1:]
    x_val = normalize(x_val, mean, std, max_len)[:, :, 1:]

    y_tr = y_tr[:, None]
    y_val = y_val[:, None]

    return x_tr, x_val, y_tr, y_val

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

        letter_idx = defaultdict(int)
        # Idx of letter in feature vector
        idx_tmp = [('A',1) , ('C',2), ('T',3), ('G',4)]

        for k, v in idx_tmp:
            letter_idx[k] = v

        for row in tsv:

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
                
                ## add name -> idx to dict
                if row[0] not in name_to_id:
                    name_to_id[row[0]] = idx
                    idx += 1

                feat_contig.append(np.concatenate(feat, 0))

                # add features
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

            feat.append(np.array(5 * [0] + [int(ri) for ri in row[4:(w_chimera - 2)]])[None, :].astype(np.uint8))
            feat[-1][0][letter_idx[row[3]]] = 1


    # Save processed data into pickle file
    print("Number of processed labels: %d" % len(target_contig_ext))
    with open(os.path.join(data_path, features_out), 'wb') as f:
        pickle.dump([feat_contig, target_contig, target_contig_edit, target_contig_ext, name_to_id], f)

def pickle_data_b(data_path, features_in, features_out):
    """
    One time function parsing the csv file and dumping the 
    values of interest into a pickle file. 
    """
    feat_contig, target_contig = [], []
    name_to_id = {}

    # Dictionary for one-hot encoding
    letter_idx = defaultdict(int)
    # Idx of letter in feature vector
    idx_tmp = [('A',1) , ('C',2), ('T',3), ('G',4)]

    for k, v in idx_tmp:
        letter_idx[k] = v

    idx = 0
    #Read tsv and process features
    with gzip.open(os.path.join(data_path, features_in), 'rt') as f:

        tsv = csv.reader(f, delimiter='\t')
        col_names = next(tsv)

        w_ext = col_names.index('Extensive_misassembly')
        w_chimera = col_names.index('chimeric')

        for row in tsv:
            name_contig = row[0]

            # If name not in set, add previous contig and target to dataset
            if name_contig not in name_to_id:
                if idx != 0:
                    feat_contig.append(np.concatenate(feat, 0))
                    target_contig.append(float(tgt))

                feat = []
               
                #Set target
                tgt = row[w_ext]
                if tgt == '':
                    tgt = 0
                else:
                    tgt = 1

                name_to_id[name_contig] = idx
                idx += 1

            # Feature vec
            feat.append(np.array(5 * [0] + [int(ri) for ri in row[4:(w_chimera - 2)]])[None, :].astype(np.uint8))
            feat[-1][0][letter_idx[row[3]]] = 1

    # Append last
    feat_contig.append(np.concatenate(feat, 0))
    target_contig.append(float(tgt))

    assert(len(feat_contig) == len(name_to_id))

    # Save processed data into pickle file
    with open(os.path.join(data_path, features_out), 'wb') as f:
        pickle.dump([feat_contig, target_contig, name_to_id], f)

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

def reverse_dict(d):

    r_d = {}
    for k, v in d.items():
        if v not in r_d:
            r_d[v] = [k]
        else:
            r_d[v].append(k)
    return r_d





