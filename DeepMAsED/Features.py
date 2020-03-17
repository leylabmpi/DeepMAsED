#!/usr/bin/env python
from __future__ import print_function
# batteries
import os
import sys
import gzip
import argparse
import logging
import itertools
from distutils.spawn import find_executable
from functools import partial
from subprocess import Popen, PIPE
from multiprocessing import Pool
# 3rd party
import pysam


IDX = {'A':0, 'C':1, 'G':2, 'T':3, 'N':-1}

def count_SNPs(query_seqs, ref_seq):
    """ Counting SNPs for all reads mapped to a position on the ref sequence
    """
    SNP_cnt = 0
    for i,x in enumerate(query_seqs):
        if i != IDX[ref_seq]:
            SNP_cnt += x[0]
    return SNP_cnt

def _contig_stats(contig, bam_file, fasta_file, assembler):
    """ getting status for reads mapped to a contig
    """
    fasta = pysam.FastaFile(fasta_file)
    
    x = 'rb' if bam_file.endswith('.bam') else 'r'
    
    with pysam.AlignmentFile(bam_file, x) as inF:
        stats = []
        contig_i = inF.references.index(contig)
        for pos in range(0,inF.lengths[contig_i]):
            # ref seq
            ref_seq = fasta.fetch(contig, pos, pos+1)
            # query seq
            query_seq = inF.count_coverage(contig, pos, pos+1)
            # SNPs
            SNPs = count_SNPs(query_seq, ref_seq)
            # coverage
            coverage = sum([x[0] for x in query_seq])
            # reads
            n_discord = 0
            n_sup = 0
            n_sec = 0
            for read in inF.fetch(contig, pos, pos+1):
                ## discordant reads
                if (read.is_paired == True and
                    read.is_proper_pair == False and
                    read.is_unmapped == False and
                    read.mate_is_unmapped == False):
                    n_discord += 1
                ## sup/sec reads
                if read.is_supplementary:
                    n_sup += 1
                if read.is_secondary:
                    n_sec += 1
            
            # columns
            stats.append([
                assembler,
                contig,
                'NA',
                str(pos),
                ref_seq,
                str(query_seq[0][0]),
                str(query_seq[1][0]),
                str(query_seq[2][0]),
                str(query_seq[3][0]),
                str(SNPs),
                str(coverage),
                str(n_discord),
                str(n_sup),
                str(n_sec),
            ])
        return stats

def contig_stats(contigs, bam_file, fasta_file, assembler):
    stats = []
    for contig in contigs:
        x = _contig_stats(contig, bam_file, fasta_file, assembler)
        stats.append(x)
    return stats

def batch_contigs(contigs, nprocs):
    msg = 'Batching contigs into {} equal bins'
    logging.info(msg.format(nprocs))
    
    contig_bins = {}
    for contig,_bin in zip(contigs, itertools.cycle(range(0,nprocs))):
        try:
            contig_bins[_bin].append(contig)
        except KeyError:
            contig_bins[_bin] = [contig]
    return contig_bins.values()

def bam_to_feats(bam_file, fasta_file, outF, assembler='NA', procs=1, debug=False):
    """ Converting bam + ref_contig_fasta to feature table
    """
    # header
    H = ['assembler', 'contig', 'Extensive_misassembly',
         'position', 'ref_base',
         'num_query_A', 'num_query_C', 'num_query_G', 'num_query_T',
         'num_SNPs', 'coverage', 'num_discordant',
         'num_supplementary', 'num_secondary']
    try:
        outF.write('\t'.join(H) + '\n')
    except TypeError:
        outF.write(('\t'.join(H) + '\n').encode())
    
    # Getting contig list
    x = 'rb' if bam_file.endswith('.bam') else 'r'
    contigs = []
    with pysam.AlignmentFile(bam_file, x) as inF:
        contigs = inF.references
    msg = 'Number of contigs in the bam file: {}'
    logging.info(msg.format(len(contigs)))

    # debug
    if debug:
        contigs = contigs[:3]
        
    # getting contig stats
    func = partial(contig_stats, bam_file=bam_file,
                   fasta_file=fasta_file, assembler=assembler)
    if procs > 1:
        p = Pool(procs)
        # batching contigs for multiprocessing
        contig_bins = batch_contigs(contigs, procs)
        # getting stats
        stats = p.map(func, contig_bins)
    else:
        # getting status
        stats = map(func, [contigs])
        
    # printing results
    for x in stats:
        for y in x:
            for z in y:
                try:
                    outF.write('\t'.join(z) + '\n')
                except TypeError:
                    outF.write(('\t'.join(z) + '\n').encode())

                
def find_file(infile, alt_dir):
    """ Finding bam/fasta files in various potential directories
    """
    orig_file = infile
    if not os.path.isfile(infile):
        infile = os.path.join(alt_dir, os.path.split(infile)[1])
        if not os.path.isfile(infile):
            infile = os.path.join(os.getcwd(), os.path.split(infile)[1])
            if not os.path.isfile(infile):
                raise IOError('Cannot find file: {}'.format(orig_file))        
    return infile

def index_fasta(orig_file, file_format='fa', overwrite=True):
    """ faidx indexing with pysam
    """
    if overwrite or not has_index_file(original_file, file_format=file_format):
        logging.info('Indexing file: {}'.format(orig_file))
        if file_format.lower() == 'fa':
            pysam.faidx(orig_file)
        elif file_format.lower() == 'vcf':
            pysam.tabix_index(orig_file, preset='vcf', force=True)
        elif file_format.lower() == 'vci':
            pysam.tabix_index(orig_file, seq_col=0, start_col=1, end_col=1, force=True)
        else:
            raise G2GValueError("Unknown file format: {0}".format(file_format))

def index_bam(bam_file, threads=1):
    bai_file = bam_file + '.bai'
    exe = 'samtools'
    if not os.path.isfile(bai_file):
        logging.info('Cannot find {}; creating...'.format(bai_file))
        # samtools in path?
        if not find_executable(exe):
            msg = 'Must index bam file, but samtools not in your PATH'
            raise IOError(msg)
        # indexing
        cmd = ['samtools' , 'index', '-@', str(threads), bam_file]
        p = Popen(cmd, stdout=PIPE, stderr=PIPE)
        output, err = p.communicate()
        rc = p.returncode
        if rc != 0:
            print(output.decode('ascii'))
            print(err.decode('ascii'))
            raise IOError('samtools return code: {}'.format(rc))
        if not os.path.isfile(bai_file):
            msg = '"samtools index" did not create the file: {}'
            raise IOError(msg.format(bai_file))        
        
def parse_bam_fasta(infile):
    """ parse bam-fasta table
    """
    bam_fasta_idx = {}
    header = {}
    req_cols = set(['bam', 'fasta'])
    with open(infile) as inF:
        for i,line in enumerate(inF):
            line = line.rstrip().split('\t')
            if i == 0:
                header = {x:ii for ii,x in enumerate(line)}
                missing = list(req_cols - set(header.keys()))
                if len(missing) > 0:
                    msg = 'Cannot find columns: {}'
                    raise ValueError(msg.format(','.join(missing)))                        
            else:
                try:
                    bam = line[header['bam']]
                    fasta = line[header['fasta']]
                except KeyError:
                    msg = 'Cannot find column values in Line {}'
                    raise KeyError(msg.format(i))
                bam = find_file(bam, os.path.split(infile)[0])
                fasta = find_file(fasta, os.path.split(infile)[0])
                bam_fasta_idx[bam] = fasta                
    return bam_fasta_idx
                
def main(args):
    """ Main interface
    """
    if args.gzip:
        _openW = lambda x: gzip.open(x, 'wb')
    else:
        _openW = lambda x: open(x, 'w')
    bam_fasta_idx = parse_bam_fasta(args.bam_fasta_table)
    feat_files = []
    for bam_file,fasta_file in bam_fasta_idx.items():
        # indexing fasta
        index_fasta(fasta_file)
        # indexing bam
        index_bam(bam_file, threads=args.procs)
        # bam-to-features
        outfile = os.path.join(args.outdir, os.path.splitext(bam_file)[0] + '_feats.tsv')
        if args.gzip:
            outfile += '.gz'
        logging.info('Processing: {}'.format(bam_file))
        with _openW(outfile) as outF:
            bam_to_feats(bam_file, fasta_file, outF, assembler='NA',
                         procs=args.procs, debug=args.debug)
        logging.info('File written: {}'.format(outfile))
        feat_files.append(outfile)
    # writing feature file table
    outfile = os.path.join(args.outdir, args.name)
    with open(outfile, 'w') as outF:
        outF.write('\t'.join(['rep', 'assembler', 'feature_file']) + '\n')
        for F in feat_files:
            outF.write('\t'.join(['NA', 'NA', F]) + '\n')
    logging.info('File written: {}'.format(outfile))
            
                
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
