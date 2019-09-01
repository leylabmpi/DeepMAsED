#!/usr/bin/env python
# import
## batteries
import os
import sys
import argparse
import _pickle as pickle
## application
from DeepMAsED import Utils

parser = argparse.ArgumentParser()
parser.add_argument('--feature_path', default='data', type=str, 
                    help='where to find feature table.')

def main(args):
    if not os.path.exists(os.path.join(args.feature_path, 'features_new.pkl')):
        utils.pickle_data_feat_only(args.feature_path,
                                    'features.tsv.gz',
                                    'features_new.pkl')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
    
