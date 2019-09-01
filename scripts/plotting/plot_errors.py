#!/usr/bin/env python
# import
## batteries
import os
import _pickle as pickle
from collections import defaultdict
## 3rd party
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from upsetplot import plot

def main():
    gap = 0
    
    colors = ["#d6cab3", "#b3c9dc"]
    xticks = np.arange(0, 10, 2)
    
    name_to_idx = {}
    names = []
    current = 0
    name_to_idx = {'None' : 0,
                   'interspecies_translocation' : 0,
                   'translocation' : 1,
                   'relocation' : 2,
                   'inversion' : 3}
    
    for idx, dataset in enumerate(['train', 'test']):
    
        n2c = defaultdict(int)
        for it, tech in enumerate(['megahit', 'metaspades']):   
            for f in os.listdir('pickles'):
                if dataset in f and tech in f:
                    with open(os.path.join('pickles', f), 'rb') as pkl:
                        er = pickle.load(pkl)
                    for e in er:
                        em = ';'.join(list(set(e.split(';'))))
                        
                        n2c[em] += er[e]
    
        bool_to_series = np.zeros((len(name_to_idx) - 1, sum(list(n2c.values())[1:]))).astype(bool)
        
        last_added = 0
        for name in n2c: 
            if name == 'None':
                continue
            for i in range(n2c[name]):
                name_all = name.split(';')
                for ni in name_all:
                    bool_to_series[name_to_idx[ni], last_added + i] = True
    
            last_added += i + 1
    
    
        df = pd.DataFrame({'inter_transloc' : bool_to_series[0],
                           'transloc' : bool_to_series[1], 
                           'reloc' : bool_to_series[2], 
                           'inversion' : bool_to_series[3]})
    
        cols = df.columns.tolist()
        aggreg = df.groupby(cols).size()
        plot(aggreg)
        plt.savefig('ag' + dataset + '.pdf')
    
    exit()
    
    max_val = np.amax(list(name_to_idx.values()))
    
    fig, ax = plt.subplots(1, 2, figsize=(13, 10))
    
    for idx, dataset in enumerate(['train', 'test']):
        for it, tech in enumerate(['megahit', 'metaspades']):
            errors = defaultdict(int)
            for f in os.listdir('pickles'):
                if dataset in f and tech in f:
                    with open(os.path.join('pickles', f), 'rb') as pkl:
                        er = pickle.load(pkl)
    
                    for e in er:
                        errors[e] += er[e]
    
            values = np.zeros(max_val + 1)
            for k in errors:
                values[name_to_idx[k]] = errors[k]
    
            tks = np.arange(len(names)) + gap
            gap += 0.4
            ax[idx].barh(tks, np.log10(values), height=0.4, color=colors[it])
    
    
        ax[idx].set_xticks(xticks)
        ax[idx].set_xticklabels([r''+str(xi) for xi in xticks], fontsize=15)
    
        ax[idx].set_xlabel(r'Number of contigs (log)', fontsize=10)
    
    ax[0].set_title(r'Training', fontsize=20)
    ax[1].set_title(r'Testing', fontsize=20)
    
    ax[0].set_yticks(tks)
    ax[0].set_yticklabels(names, fontsize=15)
    
    ax[1].set_yticks([0])
    ax[1].set_yticklabels([''], fontsize=1)
    
    
    plt.tight_layout()
    plt.savefig('tmp.pdf')

    
if __name__ == '__main__':
    main()

