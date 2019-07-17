#!/usr/bin/env python
from __future__ import print_function
import sys,os
import re
import argparse
import logging
import gzip
from functools import partial


desc = 'Renaming single genome fasta'
epi = """DESCRIPTION:
Renaming genome fasta file based on taxon name.
"""
parser = argparse.ArgumentParser(description=desc,
                                 epilog=epi,
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('fasta_file', metavar='fasta_file', type=str,
                    help='Genome fasta file')
parser.add_argument('--version', action='version', version='0.0.1')

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)


def format_fasta(old_fasta):
    if old_fasta.endswith('.gz'):
        _open = partial(gzip.open, mode='rt')
    else:
        _open = open
        
    logging.info('Reading fasta file: {}'.format(old_fasta))

    regex = re.compile(r'[^A-Za-z0-9_]+')
    seq_cnt = 0
    with _open(old_fasta) as inF:
        for line in inF:
            line = line.rstrip()
            if line.startswith('>'):
                seq_cnt += 1
                line = regex.sub('_', line.lstrip('>'))
                line = '>{}_CONTIG{}'.format(line, seq_cnt)
            print(line)
            
def main(args):
    format_fasta(args.fasta_file)

    
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
