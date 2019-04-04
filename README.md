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
  * ALE (CGAL, LAP, or REAPR)
    * provide log-likelihoods based on probabilistic assumptions
  
  

