[![Travis-CI Build Status](https://travis-ci.org/leylabmpi/DeepMAsED.svg?branch=master)](https://travis-ci.org/leylabmpi/DeepMAsED)

DeepMAsED
=========

Deep learning for Metagenome Assembly Error Detection (DeepMAsED)

*"mased"*

> Middle English term: misled, bewildered, amazed, or perplexed


# Citation

Rojas-Carulla, Mateo, Ruth E. Ley, Bernhard Schoelkopf, and Nicholas D. Youngblut. 2019.
"DeepMAsED: Evaluating the Quality of Metagenomic Assemblies." bioRxiv. https://doi.org/10.1101/763813.

# Main Description

The tool is divided into two main parts:

* **DeepMAsED-SM**
  * A snakemake pipeline for:
    * generating DeepMAsED train/test datasets from reference genomes
    * creating feature tables from "real" assemblies (fasta + bam files)
* **DeepMAsED-DL**
  * A python package for misassembly detection via deep learning


# Setup

## conda 

* [If needed] Install miniconda (or anaconda)
* See the `conda create` line in the .travis.yaml file.
* If just using DeepMAsED-SM:
  * `conda create -n snakemake conda-forge::pandas bioconda::snakemake`

### Testing the DeepMAsED package (optional)

`pytest -s`

### Installing the DeepMAsED package into the conda environment

`python setup.py install`


# Usage

## DeepMAsED-SM

### Creating feature tables for genomes (MAGs)

Feature tables are fed to DeepMAsED-DL for misassembly classification.

**Input:**

* A table of reference genomes & metagenome samples
  * The table maps reference genomes to metagenomes from which they originate.
    * If MAGs created by binning, you can either combine metagenome samples, or map genomes to many metagenome samples 
  * Table format: `<Taxon>\t<Fasta>\t<Sample>\t<Read1>\t<Read2>`
     * "Taxon" = the species/strain name of the genome
     * "Fasta" = the genome (MAG) fasta file (uncompressed or gzip'ed)
     * "Sample" = the metagenome sample from which the genome originated
       * Note: the 'sample' can just be gDNA from a cultured isolate (not a metagenome)
     * "Read1" = Illumina Read1 for the sample
     * "Read2" = Illumina Read2 for the sample
* The snakemake config file (e.g., `config.yaml`). This includes:
  * Config params on MG communities
  * Config params on assemblers & parameters
  * Note: the same config is used for simulations and feature table creation

#### Running locally 

`snakemake --use-conda -j <NUMBER_OF_THREADS> --configfile <MY_CONFIG.yaml_FILE>`

#### Running on SGE cluster 

`./snakemake_sge.sh <MY_CONFIG.yaml_FILE> cluster.json <PATH_FOR_SGE_LOGS> <NUMBER_OF_PARALLEL_JOBS> [additional snakemake options]`

It should be rather easy to update the code to run on other cluster architectures.
See the following resources for help:

* [Snakemake docs on cluster config](https://snakemake.readthedocs.io/en/stable/snakefiles/configuration.html)
* [Snakemake profiles](https://github.com/Snakemake-Profiles)

#### Output

> Assuming output directory is `./output/`

* `./output/map/`
  * Metagenome assembly error ML features
* `./output/logs/`
  * Shell process log files (also see the SGE job log files)
* `./output/benchmarks/`
  * Job resource usage info

##### Features table

* **Basic info**
  * assembler
    * metagenome assembler used
  * contig
    * contig ID
  * position
    * position on the contig (bp)
  * ref_base
    * nucleotide at that position on the contig
* **Extracted from the bam file**
  * num_query_A
    * number of reads mapping to that position with 'A'
  * num_query_C
    * number of reads mapping to that position with 'C'
  * num_query_G
    * number of reads mapping to that position with 'G'
  * num_query_T
    * number of reads mapping to that position with 'T'
  * num_SNPs
    * number of SNPs at that position
  * coverage
    * number of reads mapping to that position
  * min_insert_size
    * minimum paired-end read insert size for all reads mapping to that position
  * mean_insert_size
    * mean paired-end read insert size for all reads mapping to that position
  * stdev_insert_size
    * stdev paired-end read insert size for all reads mapping to that position
  * max_insert_size
    * max paired-end read insert size for all reads mapping to that position
  * min_mapq
    * minimum read mapping quality for all reads mapping to that position
  * mean_mapq
    * mean read mapping quality for all reads mapping to that position
  * stdev_mapq
    * stdev read mapping quality for all reads mapping to that position
  * max_mapq
    * max read mapping quality for all reads mapping to that position
  * num_proper
    * number of reads mapping to that position with proper read pairing
  * num_diff_strand
    * number of reads mapping to that position where mate maps to the other strand
    * "proper" pair alignment determined by bowtie2
  * num_orphans
    * number of reads mapping to that position where the mate did not map
  * num_supplementary
    * number of reads mapping to that position where the alignment is supplementary
    * see the [samtools docs](https://samtools.github.io/hts-specs/SAMv1.pdf) for more info
  * num_secondary
    * number of reads mapping to that position where the alignment is secondary
    * see the [samtools docs](https://samtools.github.io/hts-specs/SAMv1.pdf) for more info
  * seq_window_entropy
    * sliding window contig sequence Shannon entropy
    * window size defined with the `make_features:` param in the `config.yaml` file
  * seq_window_perc_gc
    * sliding window contig sequence GC content
    * window size defined with the `make_features:` param in the `config.yaml` file
* **miniasm info**
  * chimeric
    * chimeric contig (Supplementary alignments; SAM 0x800)
  * num_hits
    * number of primary + supplementary alignments
  * query_hit_len
    * total query hit length (all alignments summed)
  * edit_dist
    * "NM" tag in minimap2 (summed for all alignments)
  * edit_dist_norm
    * edit_dist / query_hit_len
* **MetaQUAST info**
  * Extensive_misassembly
    * the "extensive misassembly" classification set by MetaQUAST

### Creating custom train/test data from reference genomes

This is useful for training DeepMAsED-DL with a custom
train/test dataset (e.g., just biome-specific taxa). 

**Input:**

* A table listing refernce genomes. Two possible formats:
  * Genome-accession: `<Taxon>\t<Accession>`
     * "Taxon" = the species/strain name
     * "Accession" = the NCBI genbank genome accession 
     * The genomes will be downloaded based on the accession
  * Genome-fasta: `<Taxon>\t<Fasta>`
     * "Taxon" = the species/strain name of the genome
     * "Fasta" = the fasta of the genome sequence
     * Use this option if you already have the genome fasta files (uncompressed or gzip'ed)
* The snakemake config file (e.g., `config.yaml`). This includes:
  * Config params on MG communities
  * Config params on assemblers & parameters

Note: the column order for the tables doesn't matter, but the column names must be exact.

#### Output

The output will the be same as for feature generation, but with extra directories:

* `./output/genomes/`
  * Reference genomes
* `./output/MGSIM/`
  * Simulated metagenomes
* `./output/assembly/`
  * Metagenome assemblies
* `./output/true_errors/`
  * Metagenome assembly errors determined by using the references


## DeepMAsED-DL

Main interface: `DeepMAsED -h`

Note: `DeepMAsED` can be run without GPUs, but it will be much slower.

### Predicting with existing model

See `DeepMAsED predict -h` 

### Training a new model

See `DeepMAsED train -h` 

### Evaluating a model

See `DeepMAsED evalulate -h`


