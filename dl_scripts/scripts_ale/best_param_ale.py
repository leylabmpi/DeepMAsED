import os
import _pickle as pickle
import numpy as np
import argparse
from sklearn.metrics import recall_score, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
import IPython
import itertools
import operator
import bisect
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def auc_from_fpr_tpr(fpr, tpr, trapezoid=False):
    inds = [i for (i, (s, e)) in enumerate(zip(fpr[: -1], fpr[1: ])) if s != e] + [len(fpr) - 1]
    fpr, tpr = fpr[inds], tpr[inds]
    area = 0
    ft = list(zip(fpr, tpr))

    for p0, p1 in zip(ft[: -1], ft[1: ]):
        area += (p1[0] - p0[0]) * ((p1[1] + p0[1]) / 2 if trapezoid else p0[1])
    return area

def get_fpr_tpr_for_thresh(fpr, tpr, thresh):
    p = bisect.bisect_left(fpr, thresh)
    fpr = fpr.copy()
    fpr[p] = thresh
    return fpr[: p + 1], tpr[: p + 1]

parser = argparse.ArgumentParser()
parser.add_argument('--assembly_path', default='data', type=str, 
                    help='Where to find feature table.')
parser.add_argument('--save_path', default='data', type=str, 
                    help='Where to save best params.')
parser.add_argument('--path_to_predictions', default='data', type=str, 
                    help='Where to find pickle from model with true labels.')
parser.add_argument('--technology', default='megahit', type=str, 
                    help='megahit or metaspades.')
args = parser.parse_args()

assemblies = os.listdir(args.assembly_path)

path = '../data/ALE/param_search_ap_train_runs_n1000_r30'
params, scores = [], []
for f in os.listdir(path):
    tech_sc, lens = [], []
    for tech in ['megahit']:
        if not tech in f:
            continue

        with open(os.path.join(path, f), 'rb') as pk:
            sc = pickle.load(pk)
        
        tech_sc.append(sc[1])
        lens.append(sc[2])

        f = f.replace('megahit', 'metaspades')

        with open(os.path.join(path, f), 'rb') as pk:
            sc = pickle.load(pk)

        tech_sc.append(sc[1])
        lens.append(sc[2])

        # Weight scores
        scores.append((tech_sc[0] * lens[0] + tech_sc[1] * lens[1]) / sum(lens)) 
        params.append(sc[0])

sort_scores = np.argsort(scores)[::-1]
thresh = params[sort_scores[0]]

with open(os.path.join(args.save_path, 'best_param_ale_train.pkl'), 'wb') as f:
    pickle.dump([params[sort_scores[0]], scores[sort_scores[0]]], f)

