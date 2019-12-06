#!/usr/bin/env python
from __future__ import print_function
import sys,os
import argparse
import logging
import pickle
import numpy as np
import functools
import multiprocessing as mp

desc = 'Unpickle DeepMAsED predictions'
epi = """DESCRIPTION:
Unpickle >=1 DeepMAsED prediction pkl file
and convert them to a tab-delimited table.
"""
parser = argparse.ArgumentParser(description=desc,
                                 epilog=epi,
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('input_file', metavar='input_file', type=str, nargs='+',
                    help='Input file(s)')
parser.add_argument('-f', '--force', action='store_true', default=False,
                    help='Force overwrite of output files (default: %(default)s)?')
parser.add_argument('-p', '--nprocs', type=int, default=1,
                    help='Number of parallel processes (default: %(default)s)?')
parser.add_argument('--version', action='version', version='0.0.1')
#For default: (default: %(default)s)

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)


def aggregate_results(x):
    n_bins = len(x)
    min_score = np.min(x)
    mean_score = np.mean(x)
    max_score = np.max(x)
    return [n_bins, min_score, mean_score, max_score]

def unpkl(input_file, force=False):
    logging.info('Processing file: {}'.format(input_file))
    
    # output file
    output_file = os.path.splitext(input_file)[0] + '.tsv'
    if os.path.exists(output_file) and force == False:
        raise IOError('Output file already exists: {}'.format(output_file))

    # unpkl, format, & write
    with open(input_file, 'rb') as inF, open(output_file, 'w') as outF:
        header = ['sim_rep', 'contig', 'y', 'n_bins', 'min_score', 'mean_score', 'max_score']
        outF.write('\t'.join(header) + '\n')
    
        x = pickle.load(inF)
        for rep in sorted(x.keys()):
            for contig,vv in x[rep].items():
                res = aggregate_results(vv['pred'])
                res = [rep, contig, vv['y']] + res
                outF.write('\t'.join([str(x) for x in res]) + '\n')
    logging.info('File written: {}'.format(output_file))


def main(args):
    F = functools.partial(unpkl, force=args.force)
    if args.nprocs > 1:
        p = mp.Pool(args.nprocs)
        p.map(F, args.input_file)
    else:
        [F(x) for x in args.input_file]
    

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
