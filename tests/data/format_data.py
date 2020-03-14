#!/usr/bin/env python
from __future__ import print_function
import sys,os
import glob
import gzip
import argparse
import logging
import random

desc = 'Copying data from DeepMAsED-SM test to DL-test data directory'
epi = """DESCRIPTION:
Simple script to convert the files in the features/
directory created by DeepMAsED-SM to data files for
running the DeepMAsED unit tests.

In order to make sure that there are enough misassemblies
in the feature tables for training, the miassemblies are
randomly assigned to contigs (random.randint(0,1))
"""
parser = argparse.ArgumentParser(description=desc,
                                 epilog=epi,
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('features_dir', metavar='features_dir', type=str,
                    help='DeepMAsED-SM features direcotry')
parser.add_argument('out_dir', metavar='out_dir', type=str,
                    help='Output directory')
parser.add_argument('-l', '--lines', type=int, default=200000,
                    help='No. of lines from each feature table to use (default: %(default)s)')
parser.add_argument('--version', action='version', version='0.0.1')

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)


def main(args):
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
        
    # formatting feature files table
    feat_file_tbl_file = os.path.join(args.features_dir, 'feature_files.tsv')
    if not os.path.isfile(feat_file_tbl_file):
        raise IOError('Cannot find {}'.format(feat_file_tbl_file))
    feat_file_tbl_out = os.path.join(args.out_dir, 'feature_files.tsv')
    with open(feat_file_tbl_file) as inF, open(feat_file_tbl_out,'w') as outF:
        header = {}
        for i,line in enumerate(inF):
            line = line.rstrip().split('\t')
            if i == 0:
                header = {x:i for i,x in enumerate(line)}         
            if i > 0:
                ii = header['feature_file']
                line[ii] = '/'.join(line[ii].split('/')[3:])
            outF.write('\t'.join(line) + '\n')
    logging.info('Feature table file written: {}'.format(feat_file_tbl_out))

        
    # formatting feature files
    P = os.path.join(args.features_dir, '*', '*', '*', '*', 'features.tsv.gz')
    feature_files = glob.glob(P)
    if len(feature_files) == 0:
        raise IOError('Cannot find feature files at {}'.format(P))
    for feat_file in feature_files:
        logging.info('Processing feature file: {}'.format(feat_file))
        out_file = os.path.join(args.out_dir, '/'.join(feat_file.split('/')[-5:]))
        out_dir = os.path.split(out_file)[0]
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        with gzip.open(feat_file, 'rb') as inF, gzip.open(out_file, 'wt') as outF:
            cur_contig = None
            last_contig = ''
            header = {}
            misasmbl = str(random.randint(0,1))   # random assignment of misassembly
            for i,line in enumerate(inF):
                line = line.decode('utf8').rstrip().split('\t')
                if i == 0:
                    header = {x:i for i,x in enumerate(line)}
                else:
                    cur_contig = line[header['contig']]
                    if cur_contig != last_contig:
                        misasmbl = str(random.randint(0,1))   # random assignment of misassembly                    
                    line[header['Extensive_misassembly']] = misasmbl
                line = '\t'.join(line) + '\n'
                outF.write(line)
                last_contig = cur_contig
                if i + 2 >= args.lines:
                    break                                               
        logging.info('Feature file written: {}'.format(out_file))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
