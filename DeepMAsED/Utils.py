# import
## batteries
import _pickle as pickle
import os
import sys
import csv
import gzip
import glob
import logging
from functools import partial
from collections import defaultdict
import multiprocessing as mp
## 3rd party
from keras import backend as K
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import IPython
## application


def nested_dict():
    return defaultdict(nested_dict)

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
    std = np.sqrt((feat_sq_sum / n_el - mean ** 2).clip(min=0))

    return mean, std

def splitall(path):
    """
    Fully split file path
    """
    path = os.path.abspath(path)
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path: 
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path:
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

def get_row_val(row, row_num, col_idx, col):
    try:
        x = row[col_idx[col]]
    except (IndexError, KeyError):
        msg = 'ERROR: Cannot find "{}" value in Row{} of feature file table'
        sys.stderr.write(msg.format(col, row_num) + '\n'); sys.exit(1)
    return str(x)

def find_pkl_file(feat_file, force_overwrite=False):
    """ Finding pkl file """
    if force_overwrite is True:
        logging.info('  --force-overwrite=True; creating pkl from tsv file')
        return 'tsv'
    else:
        logging.info('  --force-overwrite=False; searching for pkl version of tsv file')
    
    pkl = os.path.splitext(feat_file)[0]
    if pkl.endswith('.tsv'):
        pkl = os.path.splitext(feat_file)[0]
         
    if os.path.isfile(pkl):
        logging.info('Found pkl file: {}'.format(pkl))        
        msg = '  Using the existing pkl file. Set --force-overwrite=True'
        msg += ' to force-recreate the pkl file from the tsv file'
        logging.info(msg)
        return 'pkl'
    else:
        logging.info('  No pkl found. A pkl file will be created from the tsv file')
        return 'tsv'
                
def read_feature_file_table(feat_file_table, force_overwrite=False, technology='all-asmbl'):
    """ Loads feature file table, which lists all feature tables & associated
    metadata. The table is loaded based on column names
    Params:
      feat_file_table : str, file path of tsv table
      force_overwrite : bool, force create pkl files?
      technology : str, filter to just specified assembler(s)
    Returns:
      dict{file_type : {simulation_rep : {assembler : [feature_file, simulation_metadata] }}}
    """ 
    
    if feat_file_table.endswith('.gz'):
        _open = lambda x: gzip.open(x, 'rt')
    else:
        _open = lambda x: open(x, 'r')

    base_dir = os.path.split(feat_file_table)[0]
    D = nested_dict()
    with _open(feat_file_table) as f:
        # load
        tsv = csv.reader(f, delimiter='\t')
        col_names = next(tsv)
        # indexing
        colnames = ['rep', 'assembler', 'feature_file']
        colnames = {x:col_names.index(x) for x in colnames}
        
        # formatting rows
        for i,row in enumerate(tsv):
            rep = get_row_val(row, i + 2, colnames, 'rep')
            assembler = get_row_val(row, i + 2, colnames, 'assembler')
            if technology != 'all-asmbl' and assembler != technology:
                msg = 'Feature file table, Row{} => "{}" != --technology; Skipping'
                logging.info(msg.format(i+2, assembler))
            feature_file = get_row_val(row, i + 2, colnames, 'feature_file')
            if not os.path.isfile(feature_file):
                feature_file = os.path.join(base_dir, feature_file)
            if not os.path.isfile(feature_file):
                msg = 'Feature file table, Row{} => Cannot find file'
                msg += '; The file provided: {}'
                raise ValueError(msg.format(i + 2, feature_file))
            else:
                logging.info('Input file exists: {}'.format(feature_file))
                
            if feature_file.endswith('.tsv') or feature_file.endswith('.tsv.gz'):
                file_type = find_pkl_file(feature_file, force_overwrite)
            elif feature_file.endswith('.pkl'):
                file_type = 'pkl'
            else:
                msg = 'Feature file table, Row{} => file extension'
                msg += ' must be ".tsv", ".tsv.gz", or ".pkl"'
                msg += '; The file provided: {}'
                raise ValueError(msg.format(i + 2, feature_file))
            
            D[file_type][rep][assembler] = [feature_file]

    # summary
    sys.stderr.write('#-- Feature file table summary --#\n')
    n_asmbl = defaultdict(dict)
    for ft,v in D.items():
        for rep,v in v.items():
            for asmbl,v in v.items():
                try:
                    n_asmbl[ft][asmbl] += 1
                except KeyError:
                    n_asmbl[ft][asmbl] = 1
    msg = 'Assembler = {}; File type = {}; No. of files: {}\n'
    for ft,v in n_asmbl.items():
        for asmbl,v in v.items():
            sys.stderr.write(msg.format(asmbl, ft, v))
    sys.stderr.write('#--------------------------------#\n')            
            
    return D

def pickle_in_parallel(feature_files, n_procs, set_target=True):
    """
    Pickling feature files using multiproessing.Pool.
    Params:
      feature_files : list of file paths
      n_procs : int, number of parallel processes
      set_target : bool, passed to pickle_data_b()
    Returns:
      list of files
    """
    if n_procs > 1:        
        logging.info('Pickling in parallel with {} threads...'.format(n_procs))
    else:
        logging.info('Pickling...')
    pool = mp.Pool(processes = n_procs)
    # list of lists for input to pool.map
    x = []
    for rep,v in feature_files['tsv'].items():
        for asmbl,F in v.items():
            F = F[0]
            pklF = os.path.join(os.path.split(F)[0], 'features.pkl')
            x.append([F, pklF, rep, asmbl])
    # Pickle in parallel and saving file paths in dict
    func = partial(pickle_data_b, set_target=set_target)
    if n_procs > 1:
        ret = pool.map(func, x)
    else:
        ret = map(func, x)
    for y in ret:
        rep = y[2]
        asmbl = y[3]
        feature_files['pkl'][rep][asmbl] = y[1]
    return feature_files['pkl']

def load_features_tr(feat_file_table, max_len=10000, 
                     technology = None, pickle_only=False,
                     force_overwrite=False, n_procs=1):
    """
    Loads features, pre-process them and returns training. 
    Fuses data from both assemblers. 

    Inputs: 
        feat_file_path: path to the table that lists all feature files
        max_len: fixed length of contigs
        technology: assembler, megahit or metaspades.
        pickle_only: only perform pickling prior to testing. One time call. 
        force_overwrite: bool, overwrite existing files
    Outputs:
        x, y: lists, where each element comes from one metagenome, and 
          a dictionary with idx -> (metagenome, contig_name)    
    """
    # reading in feature file table
    feat_files = read_feature_file_table(feat_file_table,
                                         force_overwrite=force_overwrite,
                                         technology=technology)
    # pickling feature tables (if needed)
    feat_files = pickle_in_parallel(feat_files, n_procs, set_target=True)

    # Pre-process once if not done already
    if pickle_only:
        logging.info('--pickle-only=True; exiting')        
        exit(0)

    # for each metagenome simulation rep, combining features from each assembler together
    ## "tech" = assembler
    x, y, ye, yext, n2i = [], [], [], [], []
    for rep,v in feat_files.items():
        xtech, ytech = [], []
        for tech,filename in v.items():
            with open(filename, 'rb') as feat:
                xi, yi, n2ii = pickle.load(feat)
                xtech.append(xi)
                ytech.append(yi)

        x_in_contig, y_in_contig = [], []
        
        for xi, yi in zip(xtech, ytech):
            for j in range(len(xi)):
                len_contig = xi[j].shape[0]

                idx_chunk = 0
                while idx_chunk * max_len < len_contig:
                    chunked = xi[j][idx_chunk * max_len :
                                    (idx_chunk + 1) * max_len, :]
            
                    x_in_contig.append(chunked)
                    y_in_contig.append(yi[j])

                    idx_chunk += 1

        # Each element is a metagenome
        x.append(x_in_contig)
        yext.append(np.array(y_in_contig))

    # for binary classification
    y = yext

    return x, y

def load_features(feat_file_table, max_len=10000, 
                  technology = 'megahit', 
                  pickle_only = False,
                  force_overwrite = False,
                  n_procs = 1):
    """
    Loads features, pre-process them and returns validation data. 

    Params: 
      data_path: path to directory containing features.pkl
      max_len: fixed length of contigs
      technology: assembler, megahit or metaspades.
      pickle_only: only perform pickling prior to testing. One time call.        
    Returns:
        x, y, i2n: lists, where each element comes from one metagenome, and 
          a dictionary with idx -> (metagenome, contig_name)
    """

    # Finding feature files
    # reading in feature file table
    feat_files = read_feature_file_table(feat_file_table,
                                         force_overwrite=force_overwrite,
                                         technology=technology)
    # pickling feature tables (if needed)
    feat_files = pickle_in_parallel(feat_files, n_procs)
    if pickle_only:
        logging.info('--pickle-only=True; exiting')  
        exit()

    # loading pickled feature matrices 
    x, y, ye, yext, n2i = [], [], [], [], []
    shift = 0
    i2n_all = {}
    for rep,v in feat_files.items():
        for assembler,filename in v.items():
            with open(filename, 'rb') as feat:
                features = pickle.load(feat)
            
            xi, yi, n2ii = features        
            i2ni = reverse_dict(n2ii)
            x_in_contig, y_in_contig = [], []
            
            n2i_keys = set([])
            for j in range(len(xi)):
                len_contig = xi[j].shape[0]

                idx_chunk = 0
                while idx_chunk * max_len < len_contig:
                    chunked = xi[j][idx_chunk * max_len :
                                    (idx_chunk + 1) * max_len, :]
        
                    x_in_contig.append(chunked)
                    y_in_contig.append(yi[j])

                    i2n_all[len(x_in_contig) - 1 + shift] = (int(rep), i2ni[j][0])
                    idx_chunk += 1
                    n2i_keys.add(i2ni[j][0])

            # Each element is a metagenome
            x.append(x_in_contig)
            yext.append(np.array(y_in_contig))

            #Sanity check
            assert(len(n2i_keys - set(n2ii.keys())) == 0)
            assert(len(set(n2ii.keys()) - n2i_keys) == 0)

            shift = len(i2n_all)

    # for binary classification
    y = yext

    return x, y, i2n_all

def load_features_nogt(feat_file_table, max_len=10000, 
                       pickle_only=False,
                       force_overwrite=False,
                       n_procs=1):
    """
    Loads features for real datasets. Filters contigs with low coverage.
    WARNING: `coverage` column assumed to be second-from-last column.

    Params: 
      feat_file_table: str, path to feature file table
      max_len: str, fixed length of contigs
    Returns:
      x, y, i2n: lists, where each element comes from one metagenome, and 
          a dictionary with idx -> (metagenome, contig_name)
    """
    # reading in feature file table
    feat_files = read_feature_file_table(feat_file_table,
                                         force_overwrite=force_overwrite)
    # pickling feature tables (if needed)
    feat_files = pickle_in_parallel(feat_files, n_procs, set_target=False)
    if pickle_only:
        logging.info('--pickle-only=True; exiting')
        exit()    
        
    # loading pickled feature tables
    x, y, ye, yext, n2i = [], [], [], [], []
    shift = 0
    i2n_all = {}
    idx_coverage = -2
    i = 0
    for rep,v in feat_files.items():
        for assembler,filename in v.items():            
            # loading pickled features
            with open(filename, 'rb') as inF:
                logging.info('Loading file: {}'.format(filename))
                features = pickle.load(inF)

            # unpacking
            try:
                xi, n2ii = features
                yi = [-1 for i in range(len(xi))]
            except ValueError:
                xi, yi, n2ii = features                

            # reverse dict
            i2ni = reverse_dict(n2ii)

            # contigs
            n_contigs_filtered = 0
            x_in_contig, y_in_contig = [], []
            n2i_keys = set([])
            for j in range(len(xi)):
                len_contig = xi[j].shape[0]
                
                #Filter low coverage
                if np.amin(xi[j][:, idx_coverage]) == 0:
                    n_contigs_filtered += 1
                    continue

                idx_chunk = 0
                while idx_chunk * max_len < len_contig:
                    chunked = xi[j][idx_chunk * max_len :
                                    (idx_chunk + 1) * max_len, :]
            
                    x_in_contig.append(chunked)
                    y_in_contig.append(yi[j])

                    i2n_all[len(x_in_contig) - 1 + shift] = (i, i2ni[j][0])
                    idx_chunk += 1
                    n2i_keys.add(i2ni[j][0])
            # status
            msg = 'Contigs filtered due to low coverage: {}'
            logging.info(msg.format(n_contigs_filtered))
                    
            # Each element is a metagenome
            x.append(x_in_contig)
            yext.append(np.array(y_in_contig))
            # for next loop iteration
            shift = len(i2n_all)
            i += 1

    # for binary classification
    y = yext
    
    return x, y, i2n_all


def kfold(x, y, idx_lo, k=5):
    """Creating folds for k-fold validation
    Params:
      k : number of folds
    Returns:
      4 lists 
    """
    # check data
    if len(x) < k:
        msg = 'Number of metagenomes is < n-folds: {} < {}'
        raise IOError(msg.format(len(x), k))
    
    # Define validation data
    x_tr, y_tr = [], []
    x_val, y_val = [], []

    # setting fold lower & upper
    meta_per_fold = int(len(x) / k)
    lower = idx_lo * meta_per_fold
    upper = (idx_lo + 1) * meta_per_fold

    # creating folds
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

def pickle_data_b(x, set_target=True):
    """
    One time function parsing the csv file and dumping the 
    values of interest into a pickle file. 
    The file can be gzip'ed 
    Params:
      x : list, first 2 elements are features_in & features_out
      set_target : bool, set the target (for train) or not (for predict)?
    Returns:
      features_out       
    """
    features_in, features_out = x[:2]

    msg = 'Pickling feature data: {} => {}'
    logging.info(msg.format(features_in, features_out))

    feat_contig, target_contig = [], []
    name_to_id = {}

    # Dictionary for one-hot encoding
    letter_idx = defaultdict(int)
    # Idx of letter in feature vector
    idx_tmp = [('A',0) , ('C',1), ('T',2), ('G',3)]

    for k, v in idx_tmp:
        letter_idx[k] = v

    idx = 0
    #Read tsv and process features
    if features_in.endswith('.gz'):
        _open = lambda x: gzip.open(x, 'rt')
    else:
        _open = lambda x: open(x, 'r')            
    with _open(features_in) as f:
        # load
        tsv = csv.reader(f, delimiter='\t')
        col_names = next(tsv)
        # indexing
        w_contig = col_names.index('contig')
        w_ext = col_names.index('Extensive_misassembly')
        w_ref = col_names.index('ref_base')
        w_nA = col_names.index('num_query_A')
        w_nC = col_names.index('num_query_C')
        w_nG = col_names.index('num_query_G')
        w_nT = col_names.index('num_query_T')
        w_var = col_names.index('num_SNPs')
        w_cov = col_names.index('coverage')  # WARNING: Predict assumes coverage in -2 position
        w_dis = col_names.index('num_discordant')
        w_features = [w_nA, w_nC, w_nG, w_nT, w_var, w_cov, w_dis]
        # formatting rows
        for row in tsv:
            name_contig = row[w_contig]

            # If name not in set, add previous contig and target to dataset
            if name_contig not in name_to_id:
                if idx != 0:
                    feat_contig.append(np.concatenate(feat, 0))
                    if set_target == True:            
                        target_contig.append(float(tgt))
                feat = []
               
                #Set target (0 or 1; 1=misassembly)
                if set_target == True:
                    tgt = int(row[w_ext])
                # index
                name_to_id[name_contig] = idx
                idx += 1

            # Feature vec
            feat.append(np.array(4 * [0] + [int(row[ind]) for ind in w_features])[None, :].astype(np.uint8))
            feat[-1][0][letter_idx[row[w_ref]]] = 1

    # Append last
    feat_contig.append(np.concatenate(feat, 0))
    if set_target == True:
        target_contig.append(float(tgt))
    # Checking feature object
    assert(len(feat_contig) == len(name_to_id))

    # Save processed data into pickle file
    with open(features_out, 'wb') as f:
        logging.info('  Dumping pickle file')
        if set_target == True:
            pickle.dump([feat_contig, target_contig, name_to_id], f)
        else:
            pickle.dump([feat_contig, name_to_id], f)
            
    return x
        
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
    """Flip keys and values
    """
    r_d = {}
    for k, v in d.items():
        if v not in r_d:
            r_d[v] = [k]
        else:
            r_d[v].append(k)
    return r_d

def compute_predictions(n2i, generator, model, save_path, save_name):
    """
    Computes predictions for a model and generator, aggregating scores for long contigs.

    Inputs: 
        n2i: dictionary with contig_name -> list of idx corresponding to that contig.
        generator: deepmased data generator
    Output:
        saves scores for individual contigs
    """
    score_val = model.predict_generator(generator)

    # Compute predictions by aggregating scores for longer contigs
    score_val = score_val.flatten()
    scores = {}

    outfile = os.path.join(save_path, '_'.join([save_name, 'predictions.tsv']))
    write = open(outfile, 'w')
    csv_writer = csv.writer(write, delimiter='\t')
    csv_writer.writerow(['Collection', 'Contig', 'Deepmased_score'])
    
    for k in n2i:
        inf = n2i[k][0]
        sup = n2i[k][-1] + 1
        if k[0] not in scores:
            scores[k[0]] = {}
       
        # Make sure contig doesnt appear more than once
        assert(k[1] not in scores[k[0]])

        # Make sure we have predictions for these indices
        if sup > len(score_val):
            continue

        # Make sure all the labels for the contig coincide
        #scores[k[0]][k[1]] = {'pred' : score_val[inf : sup]}
        csv_writer.writerow([k[0], k[1], str(np.max(score_val[inf : sup]))])
    
    write.close()
    logging.info('File written: {}'.format(outfile))

def compute_predictions_y_known(y, n2i, model, dataGen):
    """
    Computes predictions for a model and generator, NOT aggregating scores for long contigs.

    Inputs: 
        n2i: dictionary with contig_name -> list of idx corresponding to that contig.
    Output:
        scores:
        pred: scores for individual contigs
        y: corresponding true labels
    """

    score_val = model.predict_generator(dataGen)

    # Compute predictions by aggregating scores for longer contigs
    score_val = score_val.flatten()
    scores = {}
    for k in n2i:
        inf = n2i[k][0]
        sup = n2i[k][-1] + 1
        if k[0] not in scores:
            scores[k[0]] = {}
       
        # Make sure contig doesnt appear more than once
        assert(k[1] not in scores[k[0]])

        # Make sure we have predictions for these indices
        if sup > len(score_val):
            continue

        # Make sure all the labels for the contig coincide
        assert((y[inf : sup] == y[inf]).all())
        scores[k[0]][k[1]] = {'y' : int(y[inf]), 'pred' : score_val[inf : sup]}

    return scores

