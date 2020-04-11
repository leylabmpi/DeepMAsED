#!/bin/bash
CWDDIR=`pwd`
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd $DIR
wget http://ftp.tue.mpg.de/ebio/nyoungblut/mg_read_data/samp1_R1.fq.gz
wget http://ftp.tue.mpg.de/ebio/nyoungblut/mg_read_data/samp1_R2.fq.gz
wget http://ftp.tue.mpg.de/ebio/nyoungblut/mg_read_data/samp2_R1.fq.gz
wget http://ftp.tue.mpg.de/ebio/nyoungblut/mg_read_data/samp2_R2.fq.gz
cd $CWDDIR
