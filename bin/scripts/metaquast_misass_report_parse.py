#!/usr/bin/env python
from __future__ import print_function
import sys,os
import re
import argparse
import logging

desc = 'Converting "contigs_report_contigs_filtered.mis_contigs.info" to a tsv table'
epi = """DESCRIPTION:
Only mis-assembly contigs will be included.

Output table columns:
*) contig ID
*) misassembly type(s)

Output is written to STDOUT
"""
parser = argparse.ArgumentParser(description=desc,
                                 epilog=epi,
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('report_file', metavar='report_file', type=str,
                    help='"contigs_report_contigs_filtered.mis_contigs.info" file created by metaQUAST')
parser.add_argument('--version', action='version', version='0.0.1')

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)


def main(args):
    # Note: the report can have multiple misassmblies per contig
    print('\t'.join(['Contig', 'Extensive_misassembly']))
    
    report = {}
    with open(args.report_file) as inF:
        contigID = None
        for line in inF:
            line = line.rstrip()
            if line.startswith('Extensive misassembly'):
                mis = line.split('(')[1].split(',')[0].split(')')[0].replace(' ', '_')
                try:
                    report[contigID].append(mis)
                except KeyError:
                    report[contigID] = [mis]
            else:
                contigID = line

    for k,v in report.items():
        print('\t'.join([k, ';'.join(v)]))
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
