import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler
import argparse
import IPython
import _pickle as pickle

from inspect import signature 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os

import itertools
import operator
import bisect

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='data', type=str, 
                    help='Where to find feature table.')
parser.add_argument('--ale_path', default='data', type=str, 
                    help='Where to find feature table.')
parser.add_argument('--save_path', default='model', type=str, 
                    help='Where to save training weights and logs.')
args = parser.parse_args()

path_to_models = os.listdir(args.save_path)
auc = []
for model_path in path_to_models:

    if not os.path.exists((os.path.join(args.save_path, model_path, 'final_model.h5'))):
        continue
    
    fig, ax = plt.subplots(1, 1, figsize=(13, 10))
    # Create plot directory   
    if not os.path.exists(os.path.join(args.save_path, model_path, 'plots', args.data_path.split('/')[-1])):
        os.makedirs(os.path.join(args.save_path, model_path, 'plots', args.data_path.split('/')[-1]))

    colors = ['#777acd', '#7aa456']
    colors_ale = ['#c65999', '#c96d44']
    for itech, tech in enumerate(['megahit', 'metaspades']):
        with open(os.path.join(args.save_path, model_path, 'predictions', 
                               args.data_path.split('/')[-1],  tech + '.pkl'), 'rb') as spred:
            scores = pickle.load(spred)

        #'---------------------------
        # ALE
        with open(os.path.join(args.ale_path, '../best_param_ale_train.pkl'), 'rb') as bestp:
            thresh, sc = pickle.load(bestp)

        assemblies = os.listdir(args.ale_path)

        #Load all
        ale_scores = []
        for mag in assemblies: 
            if not os.path.exists(os.path.join(args.ale_path, mag, tech + '_all.pkl')):
                print("Pickle file not found for " + mag)
                exit()
            with open(os.path.join(args.ale_path, mag, tech + '_all.pkl'), 'rb') as f:
                ale_scores.append(pickle.load(f))

        preds_ale, y_ale = [], []
        for i in range(len(ale_scores)):
            if int(assemblies[i]) not in scores:
                continue
            for cont in ale_scores[i]:
                if cont in scores[int(assemblies[i])]: 
                    y_ale.append(scores[int(assemblies[i])][cont]['y'])
                else:
                    # A few contigs are missing
                    continue
                total = 0
                for score in ale_scores[i][cont]:
                    total += np.sum(ale_scores[i][cont][score] < thresh[score])

                preds_ale.append(total / float(len(thresh)) / len(ale_scores[i][cont]['depth']))

        # DeepMAsED
        y, preds = [], []

        for assembly in scores:
            for contig in scores[assembly]:
                y.append(scores[assembly][contig]['y'])
                preds.append(np.mean(scores[assembly][contig]['pred']))

        auc.append(round(average_precision_score(y, preds), 3))

        precision, recall, thr = precision_recall_curve(y, preds)

        ax.step(recall, precision, color=colors[itech], alpha=1, where='post') 
      
        #Sanity check
        assert(np.sum(np.array(y) - np.array(y_ale)) == 0)
        assert(len(preds) == len(preds_ale))

        precision, recall, thr = precision_recall_curve(y_ale, preds_ale)

        ax.step(recall, precision, color=colors_ale[itech], alpha=1, where='post') 

        auc.append(round(average_precision_score(y_ale, preds_ale), 3))

        # Set ticks

        ax.set_ylim([0.0, 1.05])
        ax.set_xlim([0.0, 1.0])
        ticks = np.arange(0, 1.1, 0.2)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ticks = [r"$" + str(t) + "$" for t in ticks]
        ax.set_xticklabels(ticks, fontsize=22)
        ax.set_yticklabels(ticks, fontsize=22)
        ax.set_xlabel(r'Recall', fontsize=22)
        ax.set_ylabel(r'Precision', fontsize=22)

        ax.grid(True, axis='y', linestyle=':')
        ax.grid(True, axis='x', linestyle=':')

    #'----------------------------

    prop_pos = np.sum(y) / float(len(y))
    base = np.arange(0, 1.1, 0.1)
    av = prop_pos * np.ones(len(base))
    ax.plot([0, 1], [prop_pos, prop_pos], '--', color='gray')
    auc.append(round(prop_pos, 3))

    print("Saving PR curves...")

    plt.legend([r'DeepMAsED (MH): AP=' + str(auc[0]), r'ALE (MH): AP=' + str(auc[1]), 
                r'DeepMAsED (MS): AP=' + str(auc[2]), r'ALE (MS): AP=' + str(auc[3]), 'Random: AP=' + str(auc[4])], fontsize=17, 
                loc='upper right')

    plt.savefig(os.path.join(args.save_path, model_path, 'plots', args.data_path.split('/')[-1], 'pr_curve.pdf'), 
                bbox_inches='tight', format='pdf', dpi=5000)


