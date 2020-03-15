#!/bin/bash

CWDDIR=`pwd`
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd $DIR
wget http://ftp.tue.mpg.de/ebio/nyoungblut/genome_data/GCA_002009845.1_ASM200984v1_genomic.fna.gz
wget http://ftp.tue.mpg.de/ebio/nyoungblut/genome_data/GCA_000214495.2_ASM21449v1_genomic.fna.gz
wget http://ftp.tue.mpg.de/ebio/nyoungblut/genome_data/GCA_000764165.1_ASM76416v1_genomic.fna.gz
wget http://ftp.tue.mpg.de/ebio/nyoungblut/genome_data/GCA_900143645.1_IMG-taxon_2582580727_annotated_assembly_genomic.fna.gz
cd $CWDDIR
