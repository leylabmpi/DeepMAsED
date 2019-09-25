from __future__ import print_function
# import
## batteries
import os
import sys
import argparse
import logging
## application
from DeepMAsED import Predict

# functions
def get_desc():
    desc = 'Predict values'
    return desc

def parse_args(test_args=None, subparsers=None):
    desc = get_desc()
    epi = """DESCRIPTION:
    Predict misassemblies based on a feature table.
    The data_path must be structured as:
      training_output/
      |_ data
      |_  |_ genome
      |_     |_ metagenome
      |_         |_ features_new.pkl
      |_         |_ features.tsv.gz
      |_ deepmased.h5
      |_ mean_std_final_model.pkl
    """
    if subparsers:
        parser = subparsers.add_parser('predict', description=desc, epilog=epi,
                                       formatter_class=argparse.RawTextHelpFormatter)
    else:
        parser = argparse.ArgumentParser(description=desc, epilog=epi,
                                         formatter_class=argparse.RawTextHelpFormatter)

    # args
    parser.add_argument('data-path', metavar='data_path', type=str, 
                        help='Where to find feature table')
    parser.add_argument('save-path', metavar='model', type=str, 
                        help='Where to save training weights and logs')
    parser.add_argument('--cpu-only', action='store_true', default=False,
                        help='Only use CPUs, and no GPUs (default: %(default)s)')

    # test args
    if test_args:
        args = parser.parse_args(test_args)
        return args

    return parser

def main(args=None):
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)
    # Input
    if args is None:
        args = parse_args()
    # Main interface
    Predict.main(args)
    
# main
if __name__ == '__main__':
    pass


