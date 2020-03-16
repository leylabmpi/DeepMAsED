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
  * A snakemake pipeline for generating DeepMAsED train/test datasets from reference genomes
* **DeepMAsED-DL**
  * A python package for misassembly detection via deep learning

# Setup

## conda 

* [If needed] Install miniconda (or anaconda)
* See the `conda create` line in the [.travis.yml](./.travis.yml) file.
* If just using DeepMAsED-SM:
  * `conda create -n snakemake conda-forge::pandas bioconda::snakemake`

### Testing the DeepMAsED package (optional)

`pytest -s`

### Installing the DeepMAsED package into the conda environment

`python setup.py install`

# Usage

## Example of classifying contig misassemblies

You need to have the following input:

* fasta of metagenome assembly contigs (uncompressed)
* BAM file of metagenome reads mapped to the contigs

### Create table mapping BAM & fasta files

If multiple sets of contigs (eg., MAGs) and BAM files,
then which contigs go with which BAM files?

Create a tab-delim table of: `bam<tab>fasta` (header required)

This will be your `bam_fasta_table`, which is need for creating the features.

### Create feature table(s)

`DeepMAsED features $bam_fasta_table`

This generates >=1 feature table and a table listing all output files
(the "feature_file_table"). This feature_file_table will be the input
for `predict`

### Predict misassemblies

`DeepMAsED predict $feature_file_table`

...where `feature_filt_table` is the path to a table that lists
all feature files (see above). 

`--force-ovewrite` forces the re-creation of the pkl files, which is a bit slower
but can prevent issues.

Change `--save-path` to set the output directory.
Use `--cpu-only` to just use CPUs instead of a GPU.

#### Third, inspect the output

By default, the predictions will be written to `deepmased_predictions.tsv`.

##### Example output

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


## Creating training datasets with `DeepMAsED-SM`

This is useful for training `DeepMAsED-DL` with a custom
train/test dataset (e.g., just biome-specific taxa). 

### Input

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

> The column order for the tables doesn't matter, but the column names must be exact.

#### Running locally 

> See the "Setup" section above for snakemake installation instructions. 

`cd ./DeepMAsED-SM/`

> Edit the config.yaml file as needed (eg., changing input & output paths)

`snakemake --use-conda -j <NUMBER_OF_THREADS> --configfile <MY_CONFIG.yaml_FILE>`

#### Running on SGE cluster 

`./snakemake_sge.sh <MY_CONFIG.yaml_FILE> cluster.json <PATH_FOR_SGE_LOGS> <NUMBER_OF_PARALLEL_JOBS> [additional snakemake options]`

It should be rather easy to update the code to run on other cluster architectures.
See the following resources for help:

* [Snakemake docs on cluster config](https://snakemake.readthedocs.io/en/stable/snakefiles/configuration.html)
* [Snakemake profiles](https://github.com/Snakemake-Profiles)


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
* `./output/map/`
  * Feature tables for each simulation

## DeepMAsED-DL

Main interface: `DeepMAsED -h`

> `DeepMAsED [train|predict]` can be run without GPUs,
but the will be substantially slower.

### Predicting with existing model

See `DeepMAsED predict -h` 

### Training a new model

See `DeepMAsED train -h` 

### Evaluating a model

See `DeepMAsED evalulate -h`

### Creating features for `predict`

See `DeepMAsED features -h`

#### Features table

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
* **MetaQUAST info**
  * Extensive_misassembly
    * the "extensive misassembly" classification set by MetaQUAST
