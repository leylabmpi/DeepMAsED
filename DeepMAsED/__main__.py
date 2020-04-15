#!/usr/bin/env python
# import
from __future__ import print_function
__version__ = '0.3.1'
## batteries
import os
import sys
import argparse
## application
from DeepMAsED.Commands import Train
from DeepMAsED.Commands import Predict
from DeepMAsED.Commands import Evaluate
from DeepMAsED.Commands import Features

# funcs
def main(args=None):
    """Main entry point for application
    """
    if args is None:
        args = sys.argv[1:]
    
    desc = 'DeepMAsED: Deep learning for Metagenome Assembly Error Detection'
    epi = """DESCRIPTION:
    Usage: DeepMAsED <subcommand> <subcommand_params>
    Example: DeepMAsED train -h

    For general info, see https://github.com/leylabmpi/DeepMAsED/
    """
    parser = argparse.ArgumentParser(description=desc,
                                     epilog=epi,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--version', action='version', version=__version__)

    # subparsers
    subparsers = parser.add_subparsers()
    ## train
    train = Train.parse_args(subparsers=subparsers)
    train.set_defaults(func=Train.main)
    ## predict
    predict = Predict.parse_args(subparsers=subparsers)
    predict.set_defaults(func=Predict.main)
    ## evaluate
    evaluate = Evaluate.parse_args(subparsers=subparsers)
    evaluate.set_defaults(func=Evaluate.main)    
    ## evaluate
    features = Features.parse_args(subparsers=subparsers)
    features.set_defaults(func=Features.main)
    
    # parsing args
    if args:
        args = parser.parse_args(args)
    else:
        args = parser.parse_args()

    # running subcommands
    if len(vars(args)) > 0:
        args.func(args)
    else:
        parser.parse_args(['--help'])

    
if __name__ == '__main__':
    main()
