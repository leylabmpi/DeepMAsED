#!/usr/bin/env python
from __future__ import print_function
import sys,os
import re
import argparse
import logging
import glob
import numpy as np
import functools
import multiprocessing as mp

desc = 'Convert metaquast output to genome-contig mapping file'
epi = """DESCRIPTION:
Use metaQUAST results to map the metagenome assembly contigs
back to the corresponding reference genomes.

The output tsv table written to STDOUT.
"""
parser = argparse.ArgumentParser(description=desc,
                                 epilog=epi,
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('true_errors_dir', metavar='true_errors_dir', type=str, 
                    help='Path to DeepMAsED-SM "true_errors" output directory')
parser.add_argument('-p', '--nprocs', type=int, default=1,
                    help='Number of parallel processes (default: %(default)s)?')
parser.add_argument('--version', action='version', version='0.0.1')
#For default: (default: %(default)s)

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)



def list_mq_dirs(true_errors_dir):
    """ Listing metaquast output directories created by DeepMAsED-SM """
    dirs = glob.glob(os.path.join(true_errors_dir, '*', '*', 'quast_corrected_input'))
    logging.info('Number of metaquast output dirs found: {}'.format(len(dirs)))
    if len(dirs) < 1:
        raise IOError('No metaquast output dirs found!')
    return dirs

def list_mq_fasta_files(mq_dir):
    """ Listing genome fasta files matching pattern: 'contigs_filtered_to*.fasta' """
    files = glob.glob(os.path.join(mq_dir, 'contigs_filtered_to*.fasta'))
    logging.info('Number of fasta files found: {}'.format(len(files)))
    if len(files) < 1:
        raise IOError('No fasta files found!')
    return files

def get_contigs(fasta_file):
    """ Getting contig IDs from a genome fasta """
    logging.info('  Processing file: {}'.format(fasta_file))
    genome_id = os.path.split(fasta_file)[1]
    genome_id = os.path.splitext(genome_id)[0]
    genome_id = genome_id.split('contigs_filtered_to_')[1]
    
    contigs = []
    with open(fasta_file, 'r') as inF:
        for line in inF:
            if line.startswith('>'):
                contigs.append(line.rstrip().lstrip('>'))
    return [genome_id, contigs]

def main(args):
    print('\t'.join(['assembler', 'sim_rep', 'genome', 'contig']))    
    dirs = list_mq_dirs(args.true_errors_dir)
    for D in dirs:
        logging.info('Processing dir: {}'.format(D))
        x = os.path.split(D)[0]
        asmbl = os.path.split(x)[1]
        rep = os.path.split(os.path.split(x)[0])[1]

        fasta_files = list_mq_fasta_files(D)
        if args.nprocs > 1:
            p = mp.Pool(args.nprocs)
            genome_contig = p.map(get_contigs, fasta_files)
        else:
            genome_contig = [get_contigs(x) for x in fasta_files]

        for genome, contigs in genome_contig:
            for contig in contigs:
                print('\t'.join([asmbl, rep, genome, contig]))
        

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
