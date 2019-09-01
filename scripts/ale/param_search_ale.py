#!/usr/bin/env python
# import
## batteries
import os
import gzip
import time
import argparse
import _pickle as pickle
## 3rd party
import numpy as np
import IPython
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.metrics import average_precision_score

parser = argparse.ArgumentParser()
parser.add_argument('--assembly_path', default='data', type=str, 
                    help='Where to find feature table.')
parser.add_argument('--path_to_predictions', default='data', type=str, 
                    help='Where to find pickle from model with true labels.')
parser.add_argument('--technology', default='megahit', type=str, 
                    help='megahit or metaspades.')
parser.add_argument('--thr_len', default=-10, type=float, 
                    help='threshold for length.')
parser.add_argument('--thr_place', default=-10, type=float, 
                    help='threshold for length.')
parser.add_argument('--thr_insert', default=-10, type=float, 
                    help='threshold for length.')
parser.add_argument('--thr_kmer', default=-10, type=float, 
                    help='threshold for length.')


def main(args:
    assemblies = os.listdir(args.assembly_path)
    
    path_predictions = args.path_to_predictions
    
    if not os.path.exists(os.path.join(args.assembly_path, '../param_search_ap_' + args.assembly_path.split('/')[-1])):
        os.makedirs(os.path.join(args.assembly_path, '../param_search_ap_' + args.assembly_path.split('/')[-1]))
    
    with open(os.path.join(path_predictions, args.technology + '.pkl'), 'rb') as f:
        model_preds = pickle.load(f)
    
    #Load all
    ale_scores = []
    for mag in assemblies: 
        if not os.path.exists(os.path.join(args.assembly_path, mag,
                                           args.technology + '_all.pkl')):
            print("Pickle file not found for " + mag)
            exit()
        F = os.path.join(args.assembly_path, mag, args.technology + '_all.pkl')
        with open(F, 'rb') as f:
            ale_scores.append(pickle.load(f))
    
    thresh = {'depth': args.thr_len,
              'place' : args.thr_place,
              'insert' : args.thr_insert,
              'kmer' : args.thr_kmer}
    # Get validation and training sets
    #val_mags = assemblies[cv * lo_size : (cv + 1) * lo_size]
    #tr_mags = assemblies[0 : cv * lo_size] + assemblies[(cv + 1) * lo_size :]
    
    # Sanity check
    #assert(len(val_mags) + len(tr_mags) == len(assemblies))
    #assert(len(set(assemblies) - set(val_mags + tr_mags)) == 0) 
    
    preds, y = [], []
    for i in range(len(ale_scores)):
        if int(assemblies[i]) not in model_preds:
            continue
        for cont in ale_scores[i]:
            if cont in model_preds[int(assemblies[i])]: 
                y.append(model_preds[int(assemblies[i])][cont]['y'])
            else:
                # A few contigs are missing
                continue
            total = 0
            for score in ale_scores[i][cont]:
                total += np.sum(ale_scores[i][cont][score] < thresh[score])
    
            preds.append(total / float(len(thresh)) / len(ale_scores[i][cont]['depth']))
    
    auc = average_precision_score(y, preds)
    
    # Save best params
    idx = []
    for k in thresh:
        idx.append(str(thresh[k]))
    idx = ''.join(idx)
    
    path_to_save = os.path.join(args.assembly_path,
                                '../param_search_ap_' + args.assembly_path.split('/')[-1],
                                args.technology + idx + '.pkl')
    with open(path_to_save, 'wb') as f:
        pickle.dump([thresh, auc, len(y)], f)
    

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)


    





