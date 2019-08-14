import os
import _pickle as pickle
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import _pickle as pickle

import csv
import argparse
from scipy.stats.stats import pearsonr

which = 'train'
path_test_tsv = '../data/DeepMAsED_GTDB_genome-refs_test.tsv'
path_train_tsv = '../data/DeepMAsED_GTDB_genome-refs_'+which+'.tsv'

compl, contam = [], []

with open(path_train_tsv, 'r') as f:
    train = csv.reader(f, delimiter='\t')

    col_names = next(train)
    idx_com = col_names.index('checkm_completeness')
    idx_con = col_names.index('checkm_contamination')

    for line in train:
        compl.append(float(line[idx_com]))
        contam.append(float(line[idx_con]))

compl_tst, contam_tst = [], []

with open(path_test_tsv, 'r') as f:
    train = csv.reader(f, delimiter='\t')

    col_names = next(train)
    idx_com = col_names.index('checkm_completeness')
    idx_con = col_names.index('checkm_contamination')

    for line in train:
        compl_tst.append(float(line[idx_com]))
        contam_tst.append(float(line[idx_con]))

compl = np.array(compl)
contam = np.array(contam)

compl_tst = np.array(compl_tst)
contam_tst = np.array(contam_tst)

m = ['comp', 'cont']
leg = [r'checkM completeness', r'checkM contamination']


fig, ax = plt.subplots(1, 1, figsize=(10, 10))
to_plot = [compl, compl_tst]

ax = sns.boxplot(data=to_plot, saturation=0.3)
ax.set_xticks([0, 1])
ax.set_xticklabels([r'Negative', r'Positive'], fontsize=25)
yticks = np.linspace(round(to_plot[0].min()), round(to_plot[0].max()+0.5), 5)
ax.set_yticks(yticks)
yticks = [r"$" + str(t) + "$" for t in yticks]
ax.set_yticklabels(yticks, fontsize=25)

ax.set_ylabel(leg[0], fontsize=25)

plt.tight_layout()
plt.savefig('box_compl.pdf')


fig, ax = plt.subplots(1, 1, figsize=(10, 10))
to_plot = [contam, contam_tst]

ax = sns.boxplot(data=to_plot, saturation=0.3)
ax.set_xticks([0, 1])
ax.set_xticklabels([r'Negative', r'Positive'], fontsize=25)
yticks = np.linspace(round(to_plot[0].min()), round(to_plot[0].max()+0.5), 5)
ax.set_yticks(yticks)
yticks = [r"$" + str(t) + "$" for t in yticks]
ax.set_yticklabels(yticks, fontsize=25)

ax.set_ylabel(leg[1], fontsize=25)

plt.tight_layout()
plt.savefig('box_contam.pdf')
