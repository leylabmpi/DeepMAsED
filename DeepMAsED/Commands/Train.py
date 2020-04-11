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
    * Partition your data into train & test, and just use
      the train data for the following 
        * see feature file table description below
    * Select a grid search of hyper-parameters to consider
      (learning rate, number of layers, etc).
    * Train with kfold = 5 (for example) for each combination of 
      hyper-parameters.
    * For each combination of hyper-parameters, check scores.pkl, 
      which contains the cross validation scores, and select the 
      hyper-parameters leading to the highest average CV
    * Re-launch the whole training with `--n-folds -1` and the best 
      hyper-parameters (this is now one single run). 

    #-- Feature File Table format --#
    * DeepMAsED-SM will generate a feature file table that lists all
      feature files and their associated metadata (eg., assembler & sim-rep).
    * The table must contain the following columns:
      * `feature_file` = the path to the feature file (created by DeepMAsED-SM, see README)
        * The files can be (gzip'ed) tab-delim or pickled (see below on `--pickle-only`)
      * `rep` = the metagenome simulation replicate 
        * Set to "1" if real data
      * `assembler` = the metadata assembler

    #-- Pickled feature files --#
    DeepMAsED-SM will generate tab-delim feature tables; however,
    DeepMAsED uses formatted & pickled versions of the tab-delim feature tables.
    `DeepMAsED train` will automatically create pickled versions of the tab-delim
    tables. These pickled versions are written to the same locations as the tab-delim
    files. If the user provides tab-delim files, but DeepMAsED finds the pickled
    versions (same name, but with `pkl` for a file extension), then DeepMAsED
    will use the pickled versions, unless `--force-overwrite=True`.
    """
    if subparsers:
        parser = subparsers.add_parser('train', description=desc, epilog=epi,
                                       formatter_class=argparse.RawTextHelpFormatter)
    else:
        parser = argparse.ArgumentParser(description=desc, epilog=epi,
                                         formatter_class=argparse.RawTextHelpFormatter)

    # args
    parser.add_argument('feature_file_table',  metavar='feature_file_table', type=str, 
                        help='Table listing feature table files (see DESCRIPTION)')
    parser.add_argument('--technology', default='all-asmbl', type=str, 
                        help='Assembler name in the data_path. "all-asmbl" will use all assemblers (default: %(default)s)')    
    parser.add_argument('--save-path', default='model', type=str, 
                        help='Where to save training weights and logs (default: %(default)s)')
    parser.add_argument('--save-name', default='deepmased', type=str, 
                        help='Prefix for name in the save-path (default: %(default)s)')  
    parser.add_argument('--filters', default=8, type=int, 
                        help='N of filters for first conv layer. Then x2 (default: %(default)s)')
    parser.add_argument('--n-hid', default=50, type=int, 
                        help='N of units in fully connected layers (default: %(default)s)')
    parser.add_argument('--n-conv', default=5, type=int, 
                        help='N of conv layers (default: %(default)s)')
    parser.add_argument('--n-fc', default=3, type=int, 
                        help='N of fully connected layers (default: %(default)s)')
    parser.add_argument('--n-epochs', default=10, type=int, 
                        help='N of training epochs (default: %(default)s)')
    parser.add_argument('--max-len', default=10000, type=int, 
                        help='Max contig len, fixed input for CNN (default: %(default)s)')
    parser.add_argument('--dropout', default=0.5, type=float, 
                        help='Rate of dropout (default: %(default)s)')
    parser.add_argument('--pool-window', default=50, type=int, 
                        help='Window size for average pooling (default: %(default)s)')
    parser.add_argument('--n-folds', default=-1, type=int, 
                        help='How many folds for CV. Use "-1" to skip & pool all data for training (default: %(default)s)')
    parser.add_argument('--lr-init', default=0.001, type=float, 
                        help='Size of test set (default: %(default)s)')
    parser.add_argument('--norm-raw', default=0, type=int, 
                        help='Whether to normalize the four one-hot feature of raw (default: %(default)s)')
    parser.add_argument('--pickle-only', action='store_true', default=False,
                        help='Only pickle files (default: %(default)s)')
    parser.add_argument('--force-overwrite', action='store_true', default=False,
                        help='Force re-creation of pickle files (default: %(default)s)')
    parser.add_argument('--seed', default=12, type=int, 
                        help='Seed used for numpy.random (default: %(default)s)')
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


