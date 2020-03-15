[![Travis-CI Build Status](https://travis-ci.org/leylabmpi/DeepMAsED.svg?branch=master)](https://travis-ci.org/leylabmpi/DeepMAsED)

DeepMAsED
=========

Deep learning for Metagenome Assembly Error Detection (DeepMAsED)

*"mased"*

> Middle English term: misled, bewildered, amazed, or perplexed


# Citation

[Mineeva, Olga, Mateo Rojas-Carulla, Ruth E. Ley, Bernhard Schölkopf, and Nicholas D. Youngblut. 2020. "DeepMAsED: Evaluating the Quality of Metagenomic Assemblies." Bioinformatics , February.](https://doi.org/10.1093/bioinformatics/btaa124)

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
* See the `conda create` line in the [.travis.yml](./travis.yml) file.
* If just using DeepMAsED-SM:
  * `conda create -n snakemake conda-forge::pandas bioconda::snakemake`

### Testing the DeepMAsED package (optional)

`pytest -s`

### Installing the DeepMAsED package into the conda environment

`python setup.py install`


# Usage

**tl;dr** 

If you just want to identify missassemblies among your metagenome assembly
contigs, then see the "Workflow for predicting misassemblies among your contigs"
section below.

## DeepMAsED-SM

### Creating feature tables for genomes (MAGs)

Feature tables are fed to DeepMAsED-DL for training models and misassembly classification.
The easiest approach is to used `DeepMAsED-SM`. Alternatively, one can just use
the [bam2feat.py](./DeepMAsED-SM/bin/scripts/bam2feat.py) script to directly
create the feature tables. The snakemake pipeline just helps to parallize the run
(on a compute cluster).

**Input for DeepMAsED-SM**

* A table mapping contigs to the metagenome samples that the originate from.
  * If using MAGs, then you can either combine metagenome samples or map genomes to
    many metagenome samples
  * Table format: `<Taxon>\t<Fasta>\t<Sample>\t<Read1>\t<Read2>`
     * "Taxon" = the species/strain name of the genome (MAG)
     * "Fasta" = the genome (MAG) contig fasta file (uncompressed or gzip'ed)
     * "Sample" = the metagenome sample from which the genome (MAG) originated
       * If this column is not provided, then DeepMAsED-SM will simulate metagenomes from the user-provided genomes,
         thus creating simulation data
     * "Read1" = Illumina Read1 for the sample
       * Only needed if "Sample" provided
     * "Read2" = Illumina Read2 for the sample
       * Only needed fi "Sample" provided and reads are paired-end 
* The snakemake config file (e.g., [config.yaml](./DeepMAsED-SM/config.yaml). This includes:
  * Config params on MG communities
  * Config params on assemblers & parameters
  * Note: the same config is used for simulations and feature table creation

#### Running locally 

> See the "Setup" section above for snakemake installation instructions. 

`snakemake --use-conda -j <NUMBER_OF_THREADS> --configfile <MY_CONFIG.yaml_FILE>`

> See the `script:` section of the [.travis.yml](.travis.yml) file for a full working example (you don't need to run `pytest`).

#### Running on SGE cluster 

`./snakemake_sge.sh <MY_CONFIG.yaml_FILE> cluster.json <PATH_FOR_SGE_LOGS> <NUMBER_OF_PARALLEL_JOBS> [additional snakemake options]`

It should be rather easy to update the code to run on other cluster architectures.
See the following resources for help:

* [Snakemake docs on cluster config](https://snakemake.readthedocs.io/en/stable/snakefiles/configuration.html)
* [Snakemake profiles](https://github.com/Snakemake-Profiles)

#### Output

> Assuming output directory in the config is `./output/`

* `./output/map/`
  * Metagenome assembly error ML features
* `./output/logs/`
  * Shell process log files (also see the SGE job log files)
* `./output/benchmarks/`
  * Job resource usage info

##### Features table

Created by the [bam2feat.py](./DeepMAsED-SM/bin/scripts/bam2feat.py) script.

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
  * num_discordant
    * discordant reads according to the read mapper definition
  * num_supplementary
    * number of reads mapping to that position where the alignment is supplementary
    * see the [samtools docs](https://samtools.github.io/hts-specs/SAMv1.pdf) for more info
  * num_secondary
    * number of reads mapping to that position where the alignment is secondary
    * see the [samtools docs](https://samtools.github.io/hts-specs/SAMv1.pdf) for more info
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

This is useful for training `DeepMAsED-DL` with a custom
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

Note: `DeepMAsED` can be run without GPUs, but it will be substantially slower.

### Predicting with existing model

See `DeepMAsED predict -h` 

### Training a new model

See `DeepMAsED train -h` 

### Evaluating a model

See `DeepMAsED evalulate -h`


# Workflow for predicting misassemblies among your contigs

This is assuming that you want to run the default final model
reported in our paper ([Mineeva et al., 2020](https://doi.org/10.1093/bioinformatics/btaa124)). 

## First, create the feature table(s) for all contigs

The easiest method is to use `DeepMAsED-SM`.
See the "Creating feature tables for genomes (MAGs)" section above
for instructions on how to do this.

Alternatively, you can just directly use the [bam2feat.py](DeepMAsED-SM/bin/scripts/bam2feat.py)
script for creating the feature tables if you don't want to run `snakemake`.
If you go this route, then you will need to manually creata a `feature_file_table`;
see `DeepMAsED train -h` for a description of the format. 

## Second, predict misassemblies using the default model

To predict:

`DeepMAsED predict --force-overwrite feature_file_table`

...where `feature_filt_table` is the path to a table that lists
all feature files (see above). 

`--force-ovewrite` forces the re-creation of the pkl files, which is a bit slower
but can prevent issues.

Change `--save-path` to set the output directory.
Use `--cpu-only` to just use CPUs instead of a GPU.

## Third, inspect the output

By default, the predictions will be written to `deepmased_predictions.tsv`.

Example output:

```
Collection     Contig  Deepmased_score
0       NODE_1156_length_5232_cov_4.046938      0.0007264018
0       NODE_1563_length_3868_cov_5.851298      0.03783685
0       NODE_4288_length_1225_cov_3.235897      0.070887744
1       k141_9081       8.8751316e-05
1       k141_2594       6.720424e-05
1       k141_4878       0.0015754104
2       NODE_5204_length_1290_cov_3.283401      0.00036007166
2       NODE_2848_length_2164_cov_2.982456      0.0005029738
2       NODE_446_length_6027_cov_5.812291       0.068261534
```

See [Mineeva et al., 2020](https://doi.org/10.1093/bioinformatics/btaa124)
to help decide what score cutoff is prudent for classifying
misassembled contigs.
