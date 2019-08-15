DeepMAsED
=========

version: 0.1.2

Deep learning for Metagenome Assembly Error Detection (DeepMAsED)

*"mased"*

> Middle English term: Misled, bewildered, amazed, or perplexed


# Main Description

The tool is divided into two main parts:

* DeepMAsED-DL: deep learning for misassembly detection
* DeepMAsED-SM: a snakemake pipeline for:
  * generating DeepMAsED train/test datasets from reference genomes
  * creating feature tables from "real" assemblies (fasta + bam files)


# Setup

## DeepMAsED-DL

> TODO


## DeepMAsED-SM

* Install miniconda (or anaconda)
* Create a conda env that includes `snakemake`
  * e.g., `conda create -n snakemake_env bioconda::snakemake`
* Activate conda env
  * e.g., `conda activate snakemake_env`

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

### Output

> Assuming output directory is `./output/`

* `./output/map/`
  * Metagenome assembly error ML features
* `./output/logs/`
  * Shell process log files (also see the SGE job log files)
* `./output/benchmarks/`
  * Job resource usage info



### Miassembly classification with DeepMAsED-DL

> TODO



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

### Output

The output will the be same as for feature generation, but with extra directories:

* `./output/genomes/`
  * Reference genomes
* `./output/MGSIM/`
  * Simulated metagenomes
* `./output/assembly/`
  * Metagenome assemblies
* `./output/true_errors/`
  * Metagenome assembly errors determined by using the references



  

