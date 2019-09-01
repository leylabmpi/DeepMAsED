#!/usr/bin/env python
# import
import argparse
import os
import itertools
import operator
import bisect
## batteries
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import IPython
import _pickle as pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='data', type=str, 
                    help='Where to find feature table.')
parser.add_argument('--ale_path', default='data', type=str, 
                    help='Where to find ALE.')
parser.add_argument('--save_path', default='model', type=str, 
                    help='Where to save training weights and logs.')


def main(args):
    path_to_models = os.listdir(args.save_path)
    auc = []
    
    for model_path in path_to_models:
        if not os.path.exists((os.path.join(args.save_path, model_path, 'final_model.h5'))):
            continue
        
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        # Create plot directory   
        if not os.path.exists(os.path.join(args.save_path, model_path, 'plots',
                                           args.data_path.split('/')[-1])):
            os.makedirs(os.path.join(args.save_path, model_path, 'plots',
                                     args.data_path.split('/')[-1]))
       
        # Load thresholds computed from training data
        F = os.path.join(args.save_path, model_path, 'predictions', 'pr_from_training.pkl')
        with open(F, 'rb') as thr:
            pr, re, th = pickle.load(thr)
        with open(os.path.join(args.ale_path, '../pr_ale_from_training.pkl'), 'rb') as thr:
            pr_ale, re_ale, th_ale = pickle.load(thr)
        with open(os.path.join(args.ale_path, '../best_param_ale_train.pkl'), 'rb') as bestp:
            thresh, sc = pickle.load(bestp)
    
        # Load checkM
        with open(os.path.join(args.ale_path, '../checkm_parsed.pkl'), 'rb') as cm:
            checkM = pickle.load(cm)
    
        # Load predictions from DeepMAsED
        with open(os.path.join(args.save_path, model_path, 'predictions', 
                               args.data_path.split('/')[-1],  'unk.pkl'), 'rb') as spred:
            scores = pickle.load(spred)
    
        # Dictionary to match key of scores to keys from ALE
        map_names = {}
        for k in scores:
            for cont in scores[k]:
                prev = '_'.join(cont.split('_')[0:-1])
                post = cont.split('_')[-1]
                map_names[prev] = post
    
    
        # Predictions for ALE
        compute_ale = False
        if compute_ale: 
            y_ale, preds_ale = [], [] 
    
            # Get for ALE
            ale_scores = []
            mags_ok_ale = []
    
            for ai, f in enumerate(os.listdir(args.ale_path)):
                if not 'pkl' in f:
                    continue
                mags_ok_ale.append(f.split('.')[0].replace('_all', ''))
                with open(os.path.join(args.ale_path, f), 'rb') as f:
                    ale_scores.append(pickle.load(f))
    
            ale_parsed = {}
           
            for i in range(len(ale_scores)):
                if mags_ok_ale[i] not in scores:
                    continue
    
                ale_parsed[mags_ok_ale[i]] = set([])
    
                for cont in ale_scores[i]:
                    cont_transf = cont.replace('.', '_')
                    if cont_transf not in map_names:
                        print("hm")
                        continue
                    cont_transf = cont_transf + '_' +  map_names[cont_transf] 
    
                    if cont_transf in scores[mags_ok_ale[i]]: 
                        y_ale.append(scores[mags_ok_ale[i]][cont_transf]['y'])
                    else:
                        # A few contigs are missing
                        print("Missing contig")
                        exit()
                        continue
                    total = 0
                    for score in ale_scores[i][cont]:
                        total += np.sum(ale_scores[i][cont][score] < thresh[score])
                    ale_parsed[mags_ok_ale[i]].add(cont_transf)
    
                    preds_ale.append(total / float(len(thresh)) / len(ale_scores[i][cont]['depth']))
    
            preds_ale = np.array(preds_ale)
            high_ale = th_ale[np.where(re_ale > 0.2)[0][-1]]
            low_ale = th_ale[np.where(re_ale > 0.6)[0][-1]]
    
            print("For ALE")
            print(len(preds_ale))
            print(len(np.where(preds_ale > high_ale)[0])) 
            print(len(np.where(preds_ale > low_ale)[0])) 
    
        # Predictions for DeepMASED
        high = th[np.where(re > 0.2)[0][-1]]
        low = th[np.where(re > 0.6)[0][-1]]
        IPython.embed()
    
        y, preds, comp, cont = [], [], [], []
        
        for assembly in scores:
            if assembly not in ale_parsed:
                continue
            for contig in scores[assembly]:
                if contig not in ale_parsed[assembly]:
                    continue
                y.append(scores[assembly][contig]['y'])
                preds.append(np.mean(scores[assembly][contig]['pred']))
    
    
        preds = np.array(preds)
        print("For DeepMAsED")
        print(len(preds))
        print(len(np.where(preds > high)[0]))
        print(len(np.where(preds > low)[0]))
    
        
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

