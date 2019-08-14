import os
import gzip
import IPython
import time
import argparse
import _pickle as pickle

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--assembly_path', default='data', type=str, 
                    help='Where to find feature table.')
args = parser.parse_args()

parsed_ale = {}

with gzip.open(args.assembly_path, 'rt') as gf:
    cur_contig, cont_length = None, None
    ti = time.time()
    for i, g in enumerate(gf):
        line = g.split()
    
        if 'Reference' in line[1]:
            # Sanity check
            if not (cont_length  is None): 
                assert((parsed_ale[cur_contig]['depth'] < 10 ** 5).all())
           
            cur_contig = line[2]
            cont_length = int(line[3])

            base = 10 ** 5 * np.ones(cont_length)

            parsed_ale[cur_contig] = {'depth' : base.copy(), 'place' : base.copy(), 
                                     'insert' : base.copy(), 'kmer' : base.copy()}

            next(gf)
            idx = 0
            continue

        if cur_contig is None:
            continue

        parsed_ale[cur_contig]['depth'][idx] = float(line[3])
        parsed_ale[cur_contig]['place'][idx] = float(line[4])
        parsed_ale[cur_contig]['insert'][idx] = float(line[5])
        parsed_ale[cur_contig]['kmer'][idx] = float(line[6])
        idx += 1


save_path = args.assembly_path.replace('.txt.gz', '_all.pkl')
with open(save_path, 'wb') as f:
    pickle.dump(parsed_ale, f)
