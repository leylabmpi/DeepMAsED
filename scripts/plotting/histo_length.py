#!/usr/bin/env python
# import
## batteries
import os
import sys
import argparse
## 3rd party
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import IPython
## application
from DeepMAsED import Utils


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='data', type=str, 
                    help='Where to find feature table.')
parser.add_argument('--save_path', default='model', type=str, 
                    help='Where to save training weights and logs.')
parser.add_argument('--max_len', default=10000, type=int, 
                    help='Max contig len, fixed input for CNN.')
parser.add_argument('--mode', default='extensive', type=str, 
                    help='Chimera or edit distance.')

def main(args):
    all_lens = []
    for tech in ['megahit', 'metaspades']:
    
        lens = []
        x, y, i2n = utils.load_features(args.data_path,
                                       max_len=args.max_len,
                                       standard=1,
                                        mode = args.mode, 
                                        technology=tech,
                                        pickle_only=False)
    
        x = [item for sl in x for item in sl]
    
        n2i = utils.reverse_dict(i2n)
        for k in n2i:
            len_last = x[n2i[k][-1]].shape[0]
            lens.append(args.max_len * (len(n2i[k]) - 1) + len_last)
        all_lens.append(np.array(lens))
    
    #lens = np.array(lens)
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.hist(all_lens[0], bins=np.logspace(np.log10(all_lens[0].min()),
                                          np.log10(all_lens[0].max()), 50),
            alpha=0.4, log=True, 
            label=r'MEGAHIT')
    ax.hist(all_lens[1], bins=np.logspace(np.log10(all_lens[0].min()),
                                          np.log10(all_lens[0].max()), 50),
            alpha=0.4, log=True,
            label='MetaSPAdes')
    plt.xscale('log')
    
    ticks = np.arange(3, 7)
    yticks = np.arange(1, 7)
    
    ax.set_xticks(10 ** ticks)
    ax.set_yticks(10 ** yticks)
    
    ticks = [r"$10^{" + str(t) + "}$" for t in ticks]
    yticks = [r"$10^{" + str(t) + "}$" for t in yticks]
    
    ax.set_xticklabels(ticks, fontsize=22)
    ax.set_yticklabels(yticks, fontsize=22)
    ax.set_xlabel(r'Contig length', fontsize=28)
    ax.set_ylabel(r'Number of contigs', fontsize=28)
    plt.legend(loc='upper right', fontsize=28)
    
    plt.tight_layout()
    
    name = args.data_path.split('/')[-1]
    plt.savefig('plots/hist' + name + '.pdf')
    

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)


