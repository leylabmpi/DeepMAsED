import csv
import gzip
import numpy as np
import os
import _pickle as pickle
import IPython
import time

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, 
                    help='Path to directory containing compressed features.')
parser.add_argument('--features_in', type=str, default='features.tsv.gz', 
                    help='Name of file with compressed features.')
parser.add_argument('--features_out', type=str, default='features.pkl', 
                    help='Name of output file for features.')
args = parser.parse_args()

feat_contig, target_contig, target_contig_edit = [], [], []
name_to_id = {}

idx = 0
#Read tsv and process features
with gzip.open(os.path.join(args.data_path, args.features_in), 'rt') as f:

    tsv = csv.reader(f, delimiter='\t')

    col_names = next(tsv)
    w_chimera = col_names.index('chimeric')
    w_edit = col_names.index('edit_dist_norm')

    prev_name, tgt, tgt_ed = None, None, None
    feat = []

    for row in tsv:
        if prev_name is None: 
            prev_name = row[0]
        if tgt is None: 
            tgt = row[w_chimera]
            tgt_ed = row[w_edit]

        if row[0] != prev_name:

            prev_name = row[0]
            if tgt == '':
                tgt = None
                tgt_edit = None
                feat = []
                continue

            feat_contig.append(np.concatenate(feat, 0))

            if tgt == 'FALSE':
                target_contig.append(0)
            else:
                target_contig.append(1)

            target_contig_edit.append(float(tgt_ed))

            feat = []
            tgt = None

        feat.append(np.array([int(ri) for ri in row[4:w_chimera]])[None, :])

        if row[0] not in name_to_id:
            name_to_id[row[0]] = idx
            idx += 1

# Save processed data into pickle file
with open(os.path.join(args.data_path, args.features_out), 'wb') as f:
    pickle.dump([feat_contig, target_contig, target_contig_edit, name_to_id], f)

