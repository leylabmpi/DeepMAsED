DeepMAsED
=========

Deep learning for Metagenome Assembly Error Detection (DeepMAsED)


*"mased"*

> Middle English term: Bewildered, amazed, perplexed, or misled


# Algorithm

## Simulating metagenome assemblies (via MGSIM)

### Input 

* Genome pool: `<genome_label>\t<genome_fasta>`
* Config params on MG communities
* Config params on assemblers & parameters

### Simuating reads

* MGSIM

### Assemblies

* meta-velvet
* idba_ud
* megahit
* metaSPADes

## Detecting assembly errors

* MetaQUAST vs reference genomes
  * mapping reads to ref genomes


## Creating features for DL

* reads mapped to ref genomes
  * just use metaQUAST output?
  * use genome polishing tool to get info?
    * eg., `pilon`
  * features:
    * each base position on each contig:
      * mis-assembly?
        * based on metaQUAST
	* which type (eg., SNP, chimera, etc?)
      * coverage
      * SNP? (A->T?, T-A?, Gap->Base? (insertion), Base->Gap? (deletion), etc.)
      * discordant reads

## DL training

* Differing priors based on the assembler (and parameters)?


  

