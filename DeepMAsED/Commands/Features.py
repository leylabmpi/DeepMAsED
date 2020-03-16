from __future__ import print_function
from pkg_resources import resource_filename
# import
## batteries
import os
import sys
import argparse
import logging
## application
from DeepMAsED import Features

# functions
def get_desc():
    desc = 'Create feature tables for Predict'
    return desc

def parse_args(test_args=None, subparsers=None):
    desc = get_desc()
    epi = """DESCRIPTION:
    In order to predict misassembled contigs with DeepMAsED predict,
    one must first create the table of features used for prediction.
 
    This subcommand takes as input >=1 BAM file and the associated fasta files
    of reference contigs and converted them to a set of features for each contig.
    The input is a table that maps BAM to ref-seq fasta files. 
    The format (with header): bam<tab>fasta

    The output will be a set of tab-delim feature tables (1 per input BAM-fasta pair)
    and a table summarizing all other others (the "feature_file_table").

    Note1: for a large number of contigs, the output can be
    10's of millions of rows or larger.

    Note2: we recommend filtering out all contigs <1000 bp. 
    """
    if subparsers:
        parser = subparsers.add_parser('features', description=desc, epilog=epi,
                                       formatter_class=argparse.RawTextHelpFormatter)
    else:
        parser = argparse.ArgumentParser(description=desc, epilog=epi,
                                         formatter_class=argparse.RawTextHelpFormatter)
    # args
    parser.add_argument('bam_fasta_table', metavar='bam_fasta_file', type=str,
                        help='Tab-delim table matching BAM and ref-fasta files (see Description)')
    parser.add_argument('-o', '--outdir', type=str, default='.',
                        help='Output directory (default: %(default)s)')
    parser.add_argument('-n', '--name', type=str, default='feature_file_table.tsv',
                        help='Output feature-file table name (default: %(default)s)')
    parser.add_argument('-g', '--gzip', action='store_true', default=False,
                        help='gzip feature tables (default: %(default)s)')
    parser.add_argument('-p', '--procs', type=int, default=1,
                        help='Number of parallel processes (default: %(default)s)')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='Debug mode for testing (default: %(default)s)')
    
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
    Features.main(args)
    
# main
if __name__ == '__main__':
    pass


