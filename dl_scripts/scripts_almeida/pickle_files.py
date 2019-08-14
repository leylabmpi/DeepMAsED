import os
import _pickle as pickle

import sys
sys.path.append('../dl_code')
import utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--feature_path', default='data', type=str, 
                    help='where to find feature table.')
args = parser.parse_args()

if not os.path.exists(os.path.join(args.feature_path, 'features_new.pkl')):
    utils.pickle_data_feat_only(args.feature_path, 'features.tsv.gz', 'features_new.pkl')
