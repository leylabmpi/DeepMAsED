import csv
import gzip
import os
from collections import defaultdict

import argparse
import _pickle as pickle

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='data', type=str, help='Where to find feature table.')
parser.add_argument('--save_path', default='data', type=str, 
                    help='Where to save pickles.')
args = parser.parse_args()

errors = defaultdict(int)
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

with gzip.open(args.data_path, 'rt') as f:
    tsv = csv.reader(f, delimiter='\t')
    col_names = next(tsv)

    for line in tsv:
        name = line[0]
        if line[-1] != '':
            error = line[-1]
            errors[error] += 1
        else:
            errors['None'] += 1
        try:
            while line[0] == name:
                line = next(tsv)
        except Exception:
            break

name = args.data_path.split('/')
name = '_'.join(name[-4:-1])
with open(os.path.join(args.save_path, name + '.pkl'), 'wb') as f:  
    pickle.dump(errors, f)
