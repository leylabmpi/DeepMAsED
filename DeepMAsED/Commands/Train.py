from __future__ import print_function
# import
## batteries
import os
import sys
import argparse
import logging
## application
from DeepMAsED import Train

# functions
def parse_args(test_args=None, subparsers=None):
    desc = 'Train model'
    epi = """DESCRIPTION:
    #-- Recommended training flow --#
    * Select a grid search of hyper-parameters to consider
      (learning rate, number of layers, etc).
    * Train with kfold = 5 (for example) for each combination of 
      hyper-parameters.
    * For each combination of hyper-parameters, check scores.pkl, 
      which contains the cross validation scores, and select the 
      hyper-parameters leading to the highest average CV
    * Re-launch the whole training with `--n-folds -1` and the best 
      hyper-parameters (this is now one single run). 

    #-- Feature table input --#
    The features table files should either be tab-delim & 
    gzip'ed (output from DeepMAsED-SM) and labeled "features.tsv.gz". 
    The directory structure of the feature tables should be:

    deepmased-sm_output_dir
      |- map
          |- 1
          |  |-- assembler1
          |  |    |_ features.tsv.gz
          |  |-- assembler2
          |  |     |_ features.tsv.gz
          |  |-- assemblerN
          |       |_ features.tsv.gz
          |- 2
          |  |-- assembler1
          |  |    |_ features.tsv.gz
          |  |-- assembler2
          |  |     |_ features.tsv.gz
          |  |-- assemblerN
          |       |_ features.tsv.gz
          |- N
             |-- assembler1
             |    |_ features.tsv.gz
             |-- assembler2
             |     |_ features.tsv.gz
             |-- assemblerN
                  |_ features.tsv.gz
      
    The `--data-path` should be the base path to the feature tables,
    ("deepmased-sm_output_dir" in the example above).

    If using real data (eg., mock communities) instead of (or in addition to) simulated 
    training data, just use the same basic directory structure.
    """
    if subparsers:
        parser = subparsers.add_parser('train', description=desc, epilog=epi,
                                       formatter_class=argparse.RawTextHelpFormatter)
    else:
        parser = argparse.ArgumentParser(description=desc, epilog=epi,
                                         formatter_class=argparse.RawTextHelpFormatter)

    # args
    parser.add_argument('data_path',  metavar='data-path', type=str, 
                        help='Base path to the feature tables (see the docs)')
    parser.add_argument('--save-path', default='model', type=str, 
                        help='Where to save training weights and logs (default: %(default)s)')
    parser.add_argument('--filters', default=8, type=int, 
                        help='N of filters for first conv layer. Then x2 (default: %(default)s)')
    parser.add_argument('--n-hid', default=20, type=int, 
                        help='N of units in fully connected layers (default: %(default)s)')
    parser.add_argument('--n-conv', default=2, type=int, 
                        help='N of conv layers (default: %(default)s)')
    parser.add_argument('--n-fc', default=1, type=int, 
                        help='N of fully connected layers (default: %(default)s)')
    parser.add_argument('--n-epochs', default=50, type=int, 
                        help='N of training epochs (default: %(default)s)')
    parser.add_argument('--standard', default=1, type=int, 
                        help='Binary, whether or not to standardize the features (default: %(default)s)')
    parser.add_argument('--max-len', default=10000, type=int, 
                        help='Max contig len, fixed input for CNN (default: %(default)s)')
    parser.add_argument('--dropout', default=0.1, type=float, 
                        help='Rate of dropout (default: %(default)s)')
    parser.add_argument('--pool-window', default=40, type=int, 
                        help='Window size for average pooling (default: %(default)s)')
    parser.add_argument('--n-folds', default=5, type=int, 
                        help='How many folds for CV. Use "-1" to skip & pool all data for training (default: %(default)s)')
    parser.add_argument('--lr-init', default=0.001, type=float, 
                        help='Size of test set (default: %(default)s)')
    parser.add_argument('--mode', default='extensive', type=str,
                        choices = ['extensive','edit', 'chimera'],
                        help='Classification problem (default: %(default)s)')
    parser.add_argument('--norm-raw', default=1, type=int, 
                        help='Whether to normalize the four one-hot feature of raw (default: %(default)s)')
    parser.add_argument('--pickle-only', action='store_true', default=False,
                        help='Only pickle files (default: %(default)s)')
    parser.add_argument('--force-overwrite', action='store_true', default=False,
                        help='Force re-creation of pickle files (default: %(default)s)')
    parser.add_argument('--n-procs', default=1, type=int, 
                        help='Number of parallel processes (default: %(default)s)')
    # running test args
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
    Train.main(args)
    
# main
if __name__ == '__main__':
    pass


