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
  * output to use for labeling misassembly regions of each contig
    * contigs_reports/contigs_report_contigs.mis_contigs.info
    * contigs_reports/interspecies_translocations_by_refs_contigs.info
    * ./reads_stats/combined_reference.bed
* ALE
  * likelihood of errors
* Struct. Var. detection
  * breseq
  * Bambus 2
  * Marygold
  * Anvio

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


  

