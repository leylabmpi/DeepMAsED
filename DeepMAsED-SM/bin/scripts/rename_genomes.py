#!/usr/bin/env python
from __future__ import print_function
import sys,os
import re
import argparse
import logging
import gzip
from functools import partial
import pandas as pd


desc = 'Renaming genome fasta files'
epi = """DESCRIPTION:
Renaming genome fasta files based on taxon name.
Renamed genome fasta files will be copied to 
<genomes_dir>.
"""
parser = argparse.ArgumentParser(description=desc,
                                 epilog=epi,
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('fasta_table', metavar='fasta_table', type=str,
                    help='')
parser.add_argument('genomes_dir', metavar='genomes_dir', type=str,
                    help='')
parser.add_argument('--version', action='version', version='0.0.1')


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)


def format_fasta(old_fasta, new_fasta):
    if old_fasta == new_fasta:
        msg = 'source & destination are the same: "{}" == "{}"'
        raise ValueError(msg.format(old_fasta, new_fasta))
    
    if old_fasta.endswith('.gz'):
        _open = partial(gzip.open, mode='rt')
    else:
        _open = open
        
    logging.info('Reading fasta file: {}'.format(old_fasta))

    with _open(old_fasta) as inF, open(new_fasta, 'w') as outF:
        for line in inF:
            line = line.rstrip()
            if line.startswith('>'):
                line = re.sub('[^A-Za-z0-9_]+', '_', line.lstrip('>'))
                line = '>' + line
            outF.write(line + '\n')
            
    logging.info('Fasta file written: {}'.format(new_fasta))
            

def main(args):
    # output
    args.genomes_dir = os.path.abspath(args.genomes_dir)
    if not os.path.isdir(args.genomes_dir):
        os.makedirs(args.genomes_dir)
    
    # fasta files
    fna = pd.read_csv(args.fasta_table, sep='\t')
    for col in ['Taxon', 'Fasta']:
        if col not in fna.columns:
            msg = 'Cannot find "{}" column'
            raise ValueError(msg.format(col))
    func = lambda x: re.sub('[^A-Za-z0-9_]+', '_', x)
    fna['Taxon'] = fna['Taxon'].apply(func)
    func = lambda x: os.path.join(args.genomes_dir, x + '.fna')
    fna['Fasta_new'] = fna['Taxon'].apply(func)
    
    # genomes
    for i,x in fna.iterrows():
        format_fasta(x['Fasta'], x['Fasta_new'])

    # new table
    fna['Fasta'] = fna['Fasta_new']
    fna.drop('Fasta_new', axis=1, inplace=True)
    fna.to_csv(sys.stdout, sep='\t', index=False)

        
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
