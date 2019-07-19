DeepMAsED
=========

version: 0.1.2

Deep learning for Metagenome Assembly Error Detection (DeepMAsED)


*"mased"*

> Middle English term: Bewildered, amazed, perplexed, or misled

# Setup

## DeepMAsED-SM

Snakemake pipelines for simulating data and/or creating feature tables

* Install miniconda or anaconda
* Create a conda env that includes `snakemake`
  * eg., `conda create -n snakemake_env snakemake`
* Activate conda env
  * eg., `conda activate snakemake_env`

# Run

## DeepMAsED-SM

### Simulating data & creating feature tables

# Input

Note: the column order for the tables doesn't matter, but the column names must be exact.

* A table listing refernce genomes. Two possible formats:
  * Genome-accession: `<Taxon>\t<Accession>`
     * "Taxon" = the species/strain name
     * "Accession" = the NCBI genbank genome accession 
     * The genomes will be downloaded based on the accession
  * Genome-fasta: `<Taxon>\t<Fasta>`
     * "Taxon" = the species/strain name of the genome
     * "Fasta" = the fasta of the genome sequence
     * Use this option if you already have the genome fasta files (uncompressed or gzip'ed)
* The snakemake config file (eg., `config.yaml`). This includes:
  * Config params on MG communities
  * Config params on assemblers & parameters

### Creating feature tables for genomes

eg., features for existing MAGs from a metagenome assembly

* A table of reference genomes & metagenome samples
  * The table maps reference genomes to metagenomes from which they originate.
    * If MAGs created by binning, you can either combine metagenome samples, or map genomes to many metagenome samples 
  * Table format: `<Taxon>\t<Fasta>\t<Sample>\t<Read1>\t<Read2>`
     * "Taxon" = the species/strain name of the genome
     * "Fasta" = the genome fasta file (uncompressed or gzip'ed)
     * "Sample" = the metagenome sample from which the genome originated
       * Note: the 'sample' can just be gDNA from a cultured isolate (not a metagenome)
     * "Read1" = Illumina Read1 for the sample
     * "Read2" = Illumina Read2 for the sample
* The snakemake config file (eg., `config.yaml`). This includes:
  * Config params on MG communities
  * Config params on assemblers & parameters
  * Note: the same config is used for simulations and feature table creation



## Running locally 

`snakemake --use-conda -j <NUMBER_OF_THREADS> --configfile <MY_CONFIG.yaml_FILE>`

## Running on SGE cluster 

`./snakemake_sge.sh <MY_CONFIG.yaml_FILE> cluster.json <PATH_FOR_SGE_LOGS> <NUMBER_OF_PARALLEL_JOBS> [additional snakemake options]`


# Output

> Assuming output directory is `./output/`

## `./output/genomes/`

Reference genomes

## `./output/MGSIM/`

Simulated metagenomes

## `./output/assembly/`

Metagenome assemblies

## `./output/true_errors/`

Metagenome assembly errors determined by using the references

## `./output/map/`

Metagenome assembly error ML features (not determined with references).

The "true errors" from `./output/true_errors/` have been joined to the
ML feature table to provide the ground truth labels. 

## `./output/logs/`

Shell process log files (also see the SGE job log files)

## `./output/benchmarks/`

Job resource usage info



# Algorithm

## Simulating metagenome assemblies (via MGSIM)

### Simuating reads

* MGSIM

### Assemblies

* megahit
* metaSPADes

## Detecting 'true' assembly errors for MAGs from simulated metagenomes

* metaQUAST

## Creating features for DL

* Reads mapped to reference genomes (eg., MAGs) with bowtie2. The feature table is generated solely from the resulting bam file.


## DL training

* See the manuscript


# Similar work
  * [SuRankCo](https://doi.org/10.1186/s12859-015-0644-7)
    * random forests to rank contig quality
    * validation method:
      * Human Microbiome Project mock community
      * GAGE study bacterial assemblies
  * ALE (CGAL, LAP, or REAPR)
    * provides log-likelihoods based on probabilistic assumptions
    * [ALE paper](https://www.ncbi.nlm.nih.gov/pubmed/23303509)
      * [install](https://portal.nersc.gov/dna/RD/Adv-Seq/ALE-doc/index.html#installation)
      
    
  

