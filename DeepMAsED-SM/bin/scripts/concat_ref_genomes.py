#!/usr/bin/env python
from __future__ import print_function
import sys,os
import re
import argparse
import logging
import functools
import gzip
import pandas as pd

desc = 'Combine ref genomes in MG sample'
epi = """DESCRIPTION:
Just combining ref genomes selected to be included in the simulated metagenome sample
"""
parser = argparse.ArgumentParser(description=desc,
                                 epilog=epi,
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('fasta_table', metavar='fasta_table', type=str,
                    help='')
parser.add_argument('comm_table', metavar='comm_table', type=str,
                    help='')
parser.add_argument('genomes_dir', metavar='genomes_dir', type=str,
                    help='')
parser.add_argument('--version', action='version', version='0.0.1')
#For default: (default: %(default)s)


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)


def main(args):
    # fasta files
    fna = pd.read_csv(args.fasta_table, sep='\t')
    if not 'Taxon' in fna.columns:
        raise ValueError('Cannot find "Taxon" column in fasta_table')
    func = lambda x: re.sub('[^A-Za-z0-9_]+', '_', x)
    fna['Taxon'] = fna['Taxon'].apply(func)
    if 'Fasta' not in fna.columns:
        F = lambda x: os.path.join(args.genomes_dir, x + '.fna')
        fna['Fasta'] = fna['Taxon'].apply(F)
                
    # community
    comm = pd.read_csv(args.comm_table, sep='\t')
    comm = comm.loc[comm['Perc_rel_abund'] > 0]
    comm.drop_duplicates(subset='Taxon', inplace=True)

    # joining
    df = pd.merge(fna, comm, on='Taxon', how='inner')
    if df.shape[0] < 1:
        msg = 'No overlap in Taxon labels between the tables!'
        raise ValueError(msg)

    # genomes
    for F in df['Fasta'].tolist():
        logging.info('Writing genome: {}'.format(F))
        if not os.path.isfile(F):
            msg = 'Cannot find file: {}'
            raise IOError(msg)
        if F.endswith('.gz'):
            _open = functools.partial(gzip.open)
        else:
            _open = open
        with _open(F) as inF:
            for line in inF:
                try:
                    print(line.decode("utf-8"), end='')
                except AttributeError:
                    print(line, end='')
                    

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
