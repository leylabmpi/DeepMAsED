#!/usr/bin/env python
from __future__ import print_function
import sys,os
import argparse
import logging
import tempfile
import subprocess
import time

desc = 'Removing javareconf from conda activation'
epi = """DESCRIPTION:
The script will search within the conda env directory for 
the file: activate-r-base.sh

It will comment-out the line: `R CMD javareconf > /dev/null 2>&1 || true`
"""
parser = argparse.ArgumentParser(description=desc,
                                 epilog=epi,
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('conda_dir', metavar='conda_dir', type=str,
                    help='Conda environment base directory')
parser.add_argument('--version', action='version', version='0.0.1')

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)


def main(args):
    if not os.path.isdir(args.conda_dir):
        msg = 'Directory does not exist: {}'
        raise IOError(msg.format(args.conda_dir))

    # getting target file
    msg = 'Searching for activate-r-base.sh in: {}'
    logging.info(msg.format(args.conda_dir))
    target_file = None
    for root, dirs, files in os.walk(args.conda_dir):
        for F in files:
            if F.endswith('activate-r-base.sh'):
                target_file = os.path.join(root, F)
                break
    if target_file is None:
        msg = 'WARNING: no activate-r-base.sh file found!'
        logging.warning(msg)
        sys.exit(0)
    else:
        msg = 'File found: {}'
        logging.info(msg.format(target_file))
        
    # reading in target file and writing to temp file
    lines = []
    outF = tempfile.NamedTemporaryFile(mode='w', delete=False)
    tmp_file_name = outF.name
    outF.close()
    modified = 'no'
    with open(target_file) as inF, open(tmp_file_name, 'w') as outF:
        for line in inF:
            if line.startswith('R CMD javareconf > /dev/null 2>&1 || true'):
                line = '#' + line
                modified = 'yes'
            elif line.startswith('#R CMD javareconf > /dev/null 2>&1 || true'):
                modified = 'already'
            outF.write(line)
    if modified == 'yes':
        msg = 'Modified line: `R CMD javareconf > /dev/null 2>&1 || true`'
    elif modified == 'already':
        msg = 'Target line already is: `#R CMD javareconf > /dev/null 2>&1 || true`'
    elif modified == 'no':
        msg = 'Did not find line: `R CMD javareconf > /dev/null 2>&1 || true`'
    else:
        raise ValueError('Logic error!')
    logging.info(msg)
        

    # copying file to original (non-blocking process)
    cmd = ' '.join(['cp', '-f', tmp_file_name, target_file])
    p = subprocess.Popen(cmd, shell = True)
    while p.poll() is None:
        time.sleep(1)

    # removing temp file
    os.remove(tmp_file_name)

    # status
    msg = 'File modified: {}'
    logging.info(msg.format(target_file))

    
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
