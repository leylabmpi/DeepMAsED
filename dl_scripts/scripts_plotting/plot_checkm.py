import os
import _pickle as pickle
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import _pickle as pickle


import argparse
from scipy.stats.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='data', type=str, 
                    help='Where to find feature table.')
args = parser.parse_args()

with open(args.data_path, 'rb') as f:
    checkm = pickle.load(f)

dir_raw = '../data/Almeida/AlmeidaA2019_samples-n143_MAGs-n1519'
dir_scores_dm = '../tests/combined_final_ap_2/nfilter_8_nconv_4_lr_0.0001_dropout_0.5_pool_60_nhid_60_nfc_3/predictions/AlmeidaA2019_samples-n143_MAGs-n1519/unk.pkl'

path_to_pr = '../tests/combined_final_ap_2/nfilter_8_nconv_4_lr_0.0001_dropout_0.5_pool_60_nhid_60_nfc_3/predictions/pr_from_training.pkl'

with open(path_to_pr, 'rb') as f:
    p, r, t = pickle.load(f)
high_th = t[np.where(r > 0.6)[0][-1]]

bins = os.listdir(dir_raw)

with open(dir_scores_dm, 'rb') as f:
    scores_dm = pickle.load(f)

names, compl, cont = [], [], []
for k in checkm:
    names.append(k)
    compl.append(checkm[k]['comp'])
    cont.append(checkm[k]['cont'])

preds, compl, contam = [], [], []
for assembly in scores_dm:
    if assembly in checkm:
        for cont in scores_dm[assembly]:
                preds.append(np.mean(scores_dm[assembly][cont]['pred']) > high_th)
                compl.append(checkm[assembly]['comp'])
                contam.append(checkm[assembly]['cont'])


preds = np.array(preds).astype(int)
compl = np.array(compl)
contam = np.array(contam)
print(len(preds))

m = ['comp', 'cont']
leg = [r'checkM completeness', r'checkM contamination']
for iarr, arr in enumerate([compl, contam]):

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    to_plot = []
    to_plot.append(arr[np.where(preds == 1)[0]])
    to_plot.append(arr[np.where(preds == 0)[0]])
    print(len(to_plot[0]))
    print(len(to_plot[1]))
    ax = sns.boxplot(data=to_plot, saturation=0.3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([r'Negative', r'Positive'], fontsize=25)
    yticks = np.linspace(round(to_plot[0].min()), round(to_plot[0].max()+0.5), 5)
    ax.set_yticks(yticks)
    yticks = [r"$" + str(t) + "$" for t in yticks]
    ax.set_yticklabels(yticks, fontsize=25)

    ax.set_ylabel(leg[iarr], fontsize=25)

    plt.tight_layout()
    plt.savefig('box'+ m[iarr] +'.pdf')
exit()

m = ['comp', 'cont']
titles = [r'checkM completeness', r'checkM contamination']
num_each_extreme = 3
for measure, ali in enumerate([compl, cont]):

    compl_sort = np.argsort(ali)
    to_plot, names_plt = [], []
    count = 0
    for idx in compl_sort:
        if names[idx] not in scores_dm:
            continue
        count += 1
        scores = []
        print(ali[idx])
        for cont in scores_dm[names[idx]]:
            scores.append(np.mean(scores_dm[names[idx]][cont]['pred']))
        to_plot.append(scores)
        names_plt.append(r''+names[idx].replace('_bin', ''))
        if count == num_each_extreme:
            break

    count = 0
    for idx in compl_sort[::-1]:
        if names[idx] not in scores_dm:
            continue

        count += 1
        scores = []
        print(ali[idx])
        for cont in scores_dm[names[idx]]:
            scores.append(np.mean(scores_dm[names[idx]][cont]['pred']))
        to_plot.append(scores)
        names_plt.append(r''+names[idx].replace('_bin', ''))
        if count == num_each_extreme:
            break

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    posit = np.concatenate([np.linspace(1, 4, num=num_each_extreme),
                            np.linspace(6, 9, num=num_each_extreme)])

    violin = ax.violinplot(to_plot, positions=posit, showmedians=True)

    pos = 0
    for pc in violin['bodies']:
        if pos < num_each_extreme:
            pc.set_facecolor('#99a765')
            pc.set_edgecolor('black')
        else:
            pc.set_facecolor("#96578a")
            pc.set_edgecolor('black')
        pos += 1

    yticks = np.arange(0, np.amax([np.amax(p) for p in to_plot]) + 0.1, 0.1)
    ax.set_xticks(posit)
    ax.set_yticks(yticks)

    yticks = [r"$" + str(t) + "$" for t in yticks]
    ax.set_xticklabels(names_plt, fontsize=16)
    ax.set_yticklabels(yticks, fontsize=25)

    ax.grid(True, axis='y', linestyle=':')

    ax.set_ylabel(r'DeepMAsED score', fontsize=25)
    ax.set_title(titles[measure], fontsize=27)

    plt.tight_layout()
    plt.savefig('violin' + m[measure]+'.pdf', bbox_inches='tight', 
                format='pdf', dpi=5000)

tot_comp = []
tot_cont = []
tot_pred = []

for i, k in enumerate(bins): 
    if k not in scores_dm:
        continue
    a = []
    for cont in scores_dm[k]:
        a.append(np.mean(scores_dm[k][cont]['pred']))
    tot_pred.append(np.mean(a))
    tot_comp.append(checkm[k]['comp'])
    tot_cont.append(checkm[k]['cont'])

tot_pred = np.array(tot_pred)
tot_comp = np.array(tot_comp)
tot_cont = np.array(tot_cont)

print(pearsonr(tot_pred, tot_comp))
print(pearsonr(tot_pred, tot_cont))
