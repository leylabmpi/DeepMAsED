[![Travis-CI Build Status](https://travis-ci.org/leylabmpi/DeepMAsED.svg?branch=master)](https://travis-ci.org/leylabmpi/DeepMAsED)

DeepMAsED
=========

Deep learning for Metagenome Assembly Error Detection (DeepMAsED)

*"mased"*

> Middle English term: misled, bewildered, amazed, or perplexed


# Citation

Rojas-Carulla, Mateo, Ruth E. Ley, Bernhard Schoelkopf, and Nicholas D. Youngblut. 2019. “DeepMAsED: Evaluating the Quality of Metagenomic Assemblies.” bioRxiv. https://doi.org/10.1101/763813.

# WARNINGS

This package is currently undergoing heavy development.
The UI is not stable and can change at any time (see git log for changes).

# Main Description

The tool is divided into two main parts:

* **DeepMAsED-SM**
  * a snakemake pipeline for:
    * generating DeepMAsED train/test datasets from reference genomes
    * creating feature tables from "real" assemblies (fasta + bam files)
* **DeepMAsED-DL**
  * deep learning for misassembly detection


# Setup

## DeepMAsED-SM

* [If needed] Install miniconda (or anaconda)
* Create a conda env that includes `snakemake` & `pandas`
  * e.g., `conda create -n snakemake conda-forge::pandas bioconda::snakemake`
* To activate the conda env: `conda activate snakemake`

## DeepMAsED-DL

Either from the `environment.yaml` file:

`conda create --name deepmased --file environment.yaml`

...or just by creating a new env with the following packages:

`conda create -n deepmased tensorflow=1.10 keras tensorboard scikit-learn ipython`

Make sure to activate the correct environment when running the deep learning code:

`conda activate deepmased`

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


