# import 
from __future__ import print_function
import os
import re
import sys
import getpass
import socket
import pandas as pd

# setup
## config
configfile: 'config.yaml'
## dirs 
snake_dir = config['pipeline']['snakemake_folder']
include: snake_dir + 'bin/dirs'

## load 
### temp_folder
config['pipeline']['username'] = getpass.getuser()
config['pipeline']['email'] = config['pipeline']['username'] + '@tuebingen.mpg.de'
config['pipeline']['temp_folder'] = os.path.join(config['pipeline']['temp_folder'],
                                                 config['pipeline']['username'])
### genomes file
config['genomes_tbl'] = pd.read_csv(config['genomes_file'], sep='\t', comment='#')
#### checking format
for x in ['Taxon']:
    if x not in config['genomes_tbl'].columns:
        msg = 'Column "{}" not found in genomes file'
        print(msg.format(x))
        sys.exit(1)
func = lambda x: re.sub('[^A-Za-z0-9_]+', '_', x)
config['genomes_tbl']['Taxon'] = config['genomes_tbl']['Taxon'].apply(func)

if 'Fasta' not in config['genomes_tbl'].columns:
    F = lambda x: os.path.join(genomes_dir, x + '.fna')
    config['genomes_tbl']['Fasta'] = config['genomes_tbl']['Taxon'].apply(F)
else:
    config['params']['MGSIM']['genome_download'] = 'Skip'

## output directories
config['output_dir'] = config['output_dir'].rstrip('/') + '/'
config['tmp_dir'] = os.path.join(config['pipeline']['temp_folder'],
		                 'DeepMAsED_' + str(os.stat('.').st_ino) + '/')
if not os.path.isdir(config['tmp_dir']):
    os.makedirs(config['tmp_dir'])

# config calculated parameters
config['reps'] = [x+1 for x in range(config['params']['reps'])]
config['assemblers'] = [k for k,v in config['params']['assemblers'].items() if not v.startswith('Skip')]

## modular snakefiles
include: snake_dir + 'bin/MGSIM/Snakefile'
include: snake_dir + 'bin/assembly/Snakefile'
include: snake_dir + 'bin/true_errors/Snakefile'
include: snake_dir + 'bin/map/Snakefile'


## local rules
localrules: all
    
# rules
rule all:
    input:
        # genome fasta files
        config['genomes_tbl']['Fasta'],
	# assemblies
	expand(asmbl_dir + '{rep}/{assembler}/contigs_filtered.fasta',
	       rep = config['reps'],
	       assembler = config['assemblers']),
	# true mis-assemblies
	## minimap2
        expand(true_errors_dir + '{rep}/{assembler}/minimap2_aln.paf.gz',
	       rep = config['reps'],
	       assembler = config['assemblers']),
        expand(true_errors_dir + '{rep}/{assembler}/minimap2_aln_summary.tsv',
	       rep = config['reps'],
	       assembler = config['assemblers']),        
	## metaquast
        expand(true_errors_dir + '{rep}/{assembler}/metaquast.done',
	       rep = config['reps'],
	       assembler = config['assemblers']),
	# read mapping to contigs
	expand(map_dir + '{rep}/{assembler}.bam.bai',
	       rep = config['reps'],
	       assembler = config['assemblers']),
	# feature table
        expand(map_dir + '{rep}/{assembler}/features.tsv.gz',
	       rep = config['reps'],
	       assembler = config['assemblers'])


# notifications (only first & last N lines)
onsuccess:
    print("Workflow finished, no error")
    cmd = "(head -n 1000 {log} && tail -n 1000 {log}) | fold -w 900 | mail -s 'DeepMAsED finished successfully' " + config['pipeline']['email']
    shell(cmd)

onerror:
    print("An error occurred")
    cmd = "(head -n 1000 {log} && tail -n 1000 {log}) | fold -w 900 | mail -s 'DeepMAsED => error occurred' " + config['pipeline']['email']
    shell(cmd)
