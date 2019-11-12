#!/usr/bin/env python
from __future__ import print_function
import os
import sys
import re
import argparse

desc = 'Remove all non-ascii characters from text file'
epi = """DESCRIPTION:
Output written to STDOUT
"""

parser = argparse.ArgumentParser(description=desc,
                                 epilog=epi,
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('txt_file', metavar='txt_file', type=str,
                    help='Text file to purge')
parser.add_argument('-r', '--replace', type=str, default='',
                    help='What to replace each non-ascii with (default: %(default)s)')
parser.add_argument('--version', action='version', version='0.0.1')


def main(args):
    regex = re.compile(r'([^\x00-\x7f])')
    with open(args.txt_file) as inF:
        for i,line in enumerate(inF):
            line = line.rstrip()
            m = regex.search(line)
            if m:
                msg = 'Line {}: non-ascii character found: "{}"\n'
                sys.stderr.write(msg.format(i+1, m.group(0)))
                line = regex.sub(args.replace, line)
            print(line)
    

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
