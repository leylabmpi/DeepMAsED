from __future__ import print_function
# import
## batteries
import os
import sys
import argparse
## application
from DeepMAsED import Evaluate

# functions
def get_desc():
    desc = 'Evaluate model'
    return desc

def parse_args(test_args=None, subparsers=None):
    desc = get_desc()
    epi = """DESCRIPTION:
    Evaluate a trained model.
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
        parser = subparsers.add_parser('evaluate', description=desc, epilog=epi,
                                       formatter_class=argparse.RawTextHelpFormatter)
    else:
        parser = argparse.ArgumentParser(description=desc, epilog=epi,
                                         formatter_class=argparse.RawTextHelpFormatter)

    # args
    parser.add_argument('--data_path', default='data', type=str, 
                        help='Where to find feature table (default: %(default)s)')
    parser.add_argument('--save_path', default='model', type=str, 
                        help='Where to save training weights and logs (default: %(default)s)')
    parser.add_argument('--save_plot', default=None, type=str, 
                        help='Where to save plots (default: %(default)s)')
    parser.add_argument('--max_len', default=10000, type=int, 
                        help='Max contig len, fixed input for CNN (default: %(default)s)')
    parser.add_argument('--mode', default='chimera', type=str, 
                        help='Chimera or edit distance (default: %(default)s)')
    parser.add_argument('--technology', default='megahit', type=str, 
                        help='Megahit or Metaspades (default: %(default)s)')
    parser.add_argument('--norm_raw', default=1, type=int, 
                        help='Whether to normalize the four one-hot feature of raw (default: %(default)s)')
    parser.add_argument('--is_synthetic', default=1, type=int, 
                        help='Whether the data is synthetic and thus has ground truth (default: %(default)s)')
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
    Evaluate.main(args)
    
# main
if __name__ == '__main__':
    pass


