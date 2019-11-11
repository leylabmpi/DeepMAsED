#!/usr/bin/env python
# import
## batteries
import sys,os
import argparse
import logging
import itertools
import statistics
from math import log
from random import shuffle
from functools import partial
from collections import deque
from multiprocessing import Pool
## 3rd party
import pysam

# UI
desc = 'Creating DL features from bam file'
epi = """DESCRIPTION:
The bam file should be indexed via `samtools index`.
The fasta file should be indexed via `samtools faidx`.

The output table is written to STDOUT.
"""
parser = argparse.ArgumentParser(description=desc,
                                 epilog=epi,
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('bam_file', metavar='bam_file', type=str,
                    help='bam (or sam) file')
parser.add_argument('fasta_file', metavar='fasta_file', type=str,
                    help='Reference sequences for the bam (sam) file')
parser.add_argument('-a', '--assembler', type=str, default='unknown',
                    help='Name of metagenome assembler used to create the contigs (default: %(default)s)')
parser.add_argument('-p', '--procs', type=int, default=1,
                    help='Number of parallel processes (default: %(default)s)')
parser.add_argument('--window', type=int, default=4,
                    help='Sliding window size for sequence entropy & GC content (default: %(default)s)')
parser.add_argument('--version', action='version', version='0.0.1')


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)


IDX = {'A':0, 'C':1, 'G':2, 'T':3}

def count_SNPs(query_seqs, ref_seq):
    SNP_cnt = 0
    for i,x in enumerate(query_seqs):
        if i != IDX[ref_seq]:
            SNP_cnt += x[0]
    return SNP_cnt

def entropy(seq):
    """ Calculate Shannon entropy of sequence.
    """
    cnt = [seq.count(i) for i in 'ACGT']
    d = sum(cnt)
    ent = []
    for i in [float(i)/d for i in cnt]:
        # round corner case that would cause math domain error
        if i == 0:
            i = 1
        ent.append(i * log(i, 2))
    ent = abs(-1 * sum(ent))
    return round(ent, 3)

def gc_percent(seq):
    """ Calculate fraction of GC bases within sequence.
    """
    counts = [seq.count(i) for i in 'ACGT']
    gc = float(counts[1] + counts[2])/sum(counts)
    return round(gc, 3)

def window(seq, wsize = 4):
    """ Sliding window of sequence
    """
    it = iter(seq)
    win = deque((next(it, None) for _ in range(wsize)), maxlen=wsize)
    yield win
    append = win.append
    for e in it:
        append(e)
        yield win

def seq_entropy(seq, window_size):
    """ Calculate the sequence entropy across a sliding window
    """
    if window_size > int(len(seq)/2.0):
        window_size = int(len(seq)/2.0)
        
    ent = []
    gc = []
    # 1st half (forward)
    midpoint = int(len(seq)/2.0)
    seq_sub = seq[:midpoint + window_size - 1]
    for x in window(seq_sub, window_size):
        ent.append(entropy(x))
        gc.append(gc_percent(x))
    # 2nd half (reverse)
    seq_sub = seq[midpoint - window_size:-1]
    ent_tmp = []
    gc_tmp = []
    for x in window(seq_sub, window_size):
        ent_tmp.append(entropy(x))
        gc_tmp.append(gc_percent(x))
    ent += ent_tmp[::-1]
    gc += gc_tmp[::-1]
    
    return ent, gc
    
def _contig_stats(contig, bam_file, fasta_file, assembler, window_size):
    """ Extracting contig-specific info from bam file
    """
    logging.info('Processing contig: {}'.format(contig))

    fasta = pysam.FastaFile(fasta_file)    
    x = 'rb' if bam_file.endswith('.bam') else 'r'
    
    with pysam.AlignmentFile(bam_file, x) as inF:
        stats = []
        # contig object
        contig_i = inF.references.index(contig)
        # sequence entropy & GC content along a sliding window
        seq_ents,gc_percs = seq_entropy(fasta.fetch(contig, 0,
                                                    inF.lengths[contig_i]),
                                        window_size)
        # read alignment at each position
        for pos in range(0,inF.lengths[contig_i]):
            # ref seq
            ref_seq = fasta.fetch(contig, pos, pos+1)
            # query seq
            query_seq = inF.count_coverage(contig, pos, pos+1)
            # SNPs
            SNPs = count_SNPs(query_seq, ref_seq)
            # coverage
            coverage = sum([x[0] for x in query_seq])
            # sequence entropy
            seq_ent = seq_ents[pos]
            gc_perc = gc_percs[pos]
            # reads
            n_proper = 0
            n_diff_strand = 0
            n_orphan = 0
            n_sup = 0
            n_sec = 0
            i_sizes = []
            map_quals = []
            for read in inF.fetch(contig, pos, pos+1):
                ## discordant reads
                if read.is_paired == True and read.is_unmapped == False:
                    if (read.is_proper_pair == True and
                        read.mate_is_unmapped == False):
                        n_proper += 1
                    elif (read.mate_is_unmapped == False and
                          read.is_reverse != read.mate_is_reverse):
                        n_diff_strand += 1
                    elif read.mate_is_unmapped == True:
                        n_orphan += 1
                    ## insert size
                    i_sizes.append(abs(read.template_length))
                ## sup/sec reads
                if read.is_supplementary:
                    n_sup += 1
                if read.is_secondary:
                    n_sec += 1
                ## mapping quality
                map_quals.append(read.mapping_quality)

            # aggretation
            ## insert sizes 
            try:
                min_i_size = min(i_sizes)
                mean_i_size = round(statistics.mean(i_sizes),1)
                stdev_i_size = round(statistics.stdev(i_sizes),1)
                max_i_size = max(i_sizes)
            except ValueError:
                min_i_size = 'NA'
                mean_i_size = 'NA'
                stdev_i_size = 'NA'
                max_i_size = 'NA'
            ## mapping quality
            try:
                min_map_qual = min(map_quals)
                mean_map_qual = round(statistics.mean(map_quals),1)
                stdev_map_qual = round(statistics.stdev(map_quals),1)
                max_map_qual = max(map_quals)
            except ValueError:
                min_map_qual = 'NA'
                mean_map_qual = 'NA'
                stdev_map_qual = 'NA'
                max_map_qual = 'NA'
                
            # columns
            stats.append([
                assembler,             # assembler ID
                contig,                # contig ID
                str(pos),              # position (bp)
                ref_seq,               # base at position
                str(query_seq[0][0]),  # number of reads with 'A'
                str(query_seq[1][0]),  # number of reads with 'C'
                str(query_seq[2][0]),  # number of reads with 'G'
                str(query_seq[3][0]),  # number of reads with 'T'
                str(SNPs),             # number of SNPs (relative to base at position)
                str(coverage),         # total reads at position
                str(min_i_size),       # min insert size
                str(mean_i_size),      # mean insert size
                str(stdev_i_size),     # stdev insert size                  
                str(max_i_size),       # max insert size
                str(min_map_qual),     # min mapq size
                str(mean_map_qual),    # mean mapq size
                str(stdev_map_qual),   # stdev mapq size                      
                str(max_map_qual),     # max mapq size             
                str(n_proper),         # number of proper paired-end reads
                str(n_diff_strand),    # number of reads mapping to opposite strands
                str(n_orphan),         # number of pairs in which only 1 read maps
                str(n_sup),            # number of supplementary reads
                str(n_sec),            # number of secondary reads
                str(seq_ent),          # sliding window sequence entropy
                str(gc_perc)           # sliding window percent GC
            ])

        return stats

def contig_stats(contigs, bam_file, fasta_file, assembler, window_size):
    """ Extracting contig-specific info from all contigs
    """
    stats = []
    for contig in contigs:
        x = _contig_stats(contig, bam_file, fasta_file, assembler, window_size)
        stats.append(x)
    return stats

def batch_contigs(contigs, nprocs):
    """ Processing contigs in batches
    """
    msg = 'Batching contigs into {} equal bins'
    logging.info(msg.format(nprocs))
    
    contig_bins = {}
    contigs = list(contigs)
    shuffle(contigs)
    for contig,_bin in zip(contigs, itertools.cycle(range(0,nprocs))):
        try:
            contig_bins[_bin].append(contig)
        except KeyError:
            contig_bins[_bin] = [contig]
    return contig_bins.values()

def main(args):
    """ Main interface
    """
    # header
    H = ['assembler', 'contig', 'position', 'ref_base',
         'num_query_A', 'num_query_C', 'num_query_G', 'num_query_T',
         'num_SNPs', 'coverage',
         'min_insert_size', 'mean_insert_size', 'stdev_insert_size', 'max_insert_size',
         'min_mapq', 'mean_mapq', 'stdev_mapq', 'max_mapq',         
         'num_proper', 'num_diff_strand', 'num_orphans',
         'num_supplementary', 'num_secondary',
         'seq_window_entropy', 'seq_window_perc_gc']
    print('\t'.join(H))
    
    # Getting contig list
    x = 'rb' if args.bam_file.endswith('.bam') else 'r'
    contigs = []
    with pysam.AlignmentFile(args.bam_file, x) as inF:
        contigs = inF.references
    msg = 'Number of contigs in the bam file: {}'
    logging.info(msg.format(len(contigs)))

    # debug
    contigs = contigs[len(contigs)-10:]
    
    # getting contig stats
    func = partial(contig_stats, bam_file=args.bam_file,
                   fasta_file=args.fasta_file,
                   assembler=args.assembler,
                   window_size=args.window)
    if args.procs > 1:
        p = Pool(args.procs)
        # batching contigs for multiprocessing
        contig_bins = batch_contigs(contigs, args.procs)
        # getting stats
        stats = p.map(func, contig_bins)
    else:
        # getting status
        stats = map(func, [contigs])
        
    # printing results
    logging.info('Writing feature table to STDOUT')
    for batch in stats:
        for y in batch:
            for z in y:
                print('\t'.join(z))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
