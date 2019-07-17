#!/usr/bin/env python
from __future__ import print_function
import sys,os
import argparse
import logging
from Bio import SeqIO

desc = 'Filtering contigs by length'
epi = """DESCRIPTION:
A simple script for filtering a fasta by length.
It uses biopython.
Output is written to STDOUT
"""
parser = argparse.ArgumentParser(description=desc,
                                 epilog=epi,
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('fasta', metavar='fasta', type=str,
                    help='fasta file')
parser.add_argument('-l', '--length', type=int, default=500,
                    help='Contig length cutoff (default: %(default)s)')
parser.add_argument('--version', action='version', version='0.0.1')


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)


def main(args):
    with open(args.fasta) as inF:
        for record in SeqIO.parse(inF, 'fasta'):
            if len(record.seq) >= args.length:
                SeqIO.write([record], sys.stdout, 'fasta')
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
