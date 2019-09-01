from __future__ import print_function
# import
## batteries
import os
import sys
import argparse
## application
from DeepMAsED import Predict

# functions
def get_desc():
    desc = 'Predict values'
    return desc

def parse_args(test_args=None, subparsers=None):
    desc = get_desc()
    epi = """DESCRIPTION:
    """
    if subparsers:
        parser = subparsers.add_parser('predict', description=desc, epilog=epi,
                                       formatter_class=argparse.RawTextHelpFormatter)
    else:
        parser = argparse.ArgumentParser(description=desc, epilog=epi,
                                         formatter_class=argparse.RawTextHelpFormatter)

    # args
    parser.add_argument('--data_path', default='data', type=str, 
                    help='Where to find feature table.')
    parser.add_argument('--save_path', default='model', type=str, 
                        help='Where to save training weights and logs.')
    parser.add_argument('--cpu_only', dest='cpu_only', action='store_true')

    # test args
    if test_args:
        args = parser.parse_args(test_args)
        return args

    return parser

def main(args=None):
    # Input
    if args is None:
        args = parse_args()
    # Main interface
    Predict.main(args)
    
# main
if __name__ == '__main__':
    pass


