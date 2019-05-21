DeepMAsED
=========

version: 0.1.1

Deep learning for Metagenome Assembly Error Detection (DeepMAsED)


*"mased"*

> Middle English term: Bewildered, amazed, perplexed, or misled

# Setup

* Install miniconda or anaconda
* Setup conda
* Create a conda env that includes `snakemake`
  * eg., `conda create -n snakemake_env snakemake`
* Activate conda env
  * eg., `conda activate snakemake_env`

# Run

# Input 

* Refernce genomes. Two possible formats:
  * Genome-accession: `<genome_label>\t<genome_accession>`
     * The genomes will be downloaded based on the accession
     * Column names: `<genome_label> = Taxon`, `<genome_accession> = Accession`
  * Genome-fasta: `<genome_label>\t<genome_fasta>`
     * The genome fasta files are provided (uncompressed or gzip'ed)
     * Column names: `<genome_label> = Taxon`, `<genome_fasta> = Fasta`
* Config file (eg., `config.yaml`). This includes:
  * Config params on MG communities
  * Config params on assemblers & parameters

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

* meta-velvet
* idba_ud
* megahit
* metaSPADes

## Detecting assembly errors

* edit distance
  * min number of changes required to change contig to ref seq
  * eg., with Needleman-Wunsch algorithm
  * using minimap2
    * the "NM" tag is the edit distance
      * `NM = #mismatches + #I + #D + #ambiguous_bases`
    * what about chimeric contigs?
      * chimeric = differing sections of the contig best align to different ref genomes
      * if chimeric, automatically "bad" assembly
      * getting chimeras: `0x800` flag; also same QNAME
      * use: `--secondary=no` for minimap2
    * pysam implementation
      * `pysam.AlignedSegment`
        * `get_tag`
	* `is_supplementary` = chimeric
  * python-based methods:
    * https://pypi.org/project/nwalign/

* VALET (http://github.com/marbl/VALET)

* BLASTN
  * as used for the minFinder paper
* minimap2
  * Generates sam files, so small variants can be identified
* MetaQUAST vs reference genomes
  * mapping reads to ref genomes
  * output to use for labeling misassembly regions of each contig
    * contigs_reports/contigs_report_contigs.mis_contigs.info
    * contigs_reports/interspecies_translocations_by_refs_contigs.info
    * ./reads_stats/combined_reference.bed
* ALE
  * likelihood of errors
* SuRankCo-score
  * utilizes BLAT; provides many scores
* Struct. Var. detection
  * breseq
  * Bambus 2
  * Marygold
  * Anvio
  * SGTK
    * creates a scaffold graph
    * can have just contigs and ref genome(s) as input

## Creating features for DL

* reads mapped to ref genomes
  * just use metaQUAST output?
  * use genome polishing tool to get info?
    * eg., `pilon`
  * using pysam to generate the features:
    * each base position on each contig:
      * coverage
        * `samfile.pileup()`
	* `count_coverage()`
      * SNPs (A->T?, T-A?, Gap->Base? (insertion), Base->Gap? (deletion), etc.)
        * pysam.PileupColumn
	  * get_query_sequence
      * supplementary/secondary alignments
        * pysam.AlignedSegment
          * is_secondary()
	  * is_supplementary()
      * discordant reads
        * -F 14
	* pysam.AlignedSegment
	  * is_paired() == True & is_proper_pair != False & is_unmapped == False & mate_is_unmapped == False

## DL training

* Differing priors based on the assembler (and parameters)?



# Similar work
  * [SuRankCo](https://doi.org/10.1186/s12859-015-0644-7)
    * random forests to rank contig quality
    * validation method:
      * Human Microbiome Project mock community
      * GAGE study bacterial assemblies
  * ALE (CGAL, LAP, or REAPR)
    * provide log-likelihoods based on probabilistic assumptions
  
  

