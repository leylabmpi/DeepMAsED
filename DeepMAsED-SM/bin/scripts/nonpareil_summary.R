#!/usr/bin/env Rscript

# library
suppressPackageStartupMessages(library("Nonpareil"))

# args
args = commandArgs(trailingOnly = TRUE)

if(length(args) < 3){
   cat('Usage: nonpareil_summary.R <seq-depth> <out-base> <npo_file1> [.npo_file...]\n')
   cat('       seq-depth = target sequencing depth (eg., 1e9)\n')
   cat('       out-base = output file base name (eg., nonpareil_summary)\n')
   cat('\n')
   stop()
}

#' This function streamlines the analysis of metagenome dataset coverage
#' obtained using nonpareil. It calculates the coverage from npo files
#' produced by nonpareil and returns objects with diverse summaries.
nonpareil_analyses = function(npo_files, target_seq_depth = 3e9, out_base = 'nonpareil_summary'){
  # type checking
  target_seq_depth = as.numeric(target_seq_depth)

  # filtering out any empty files
  npo_files = npo_files[file.info(npo_files)$size > 0]

  # curves for all npo files
  sample_labels = sapply(npo_files, function(x) basename(dirname(x)))
  pdf_out = paste0(out_base, '_curve.pdf')
  pdf(pdf_out, width=11) 
  all_curves = Nonpareil.curve.batch(npo_files, plot=TRUE, labels=sample_labels)
  dev.off()

  # curve for each npo_file
  for(f in npo_files){
    sample_label = basename(dirname(f))
    pdf_out = paste0(tools::file_path_sans_ext(f), '_curve.pdf')
    pdf(pdf_out, width=11)
    each_curve = Nonpareil.curve(f, plot=TRUE, label=sample_label)
    dev.off()
  }

  # Table with results
  result_all = summary(all_curves)

  # Extract calculated coverage and summarise (in %)
  coverage = result_all[,"C"]
  coverage_percent = summary(coverage) * 100
  
  # Extract observed sequencing effort (in Gbp)
  obs_seq_effort = summary(result_all[,"LR"]) / 1e9

  # Extract sequencing effort for nearly complete coverage (in Gbp)
  # LRstar is the projected seq. effort for nearly complete coverage
  # In this case, a 95% coverage is considered nearly complete
  nearly_complete = summary(result_all[,"LRstar"]) / 1e9

  # Predict coverage for a sequencing effort of n Gbp (in %)
  predicted_coverage = sapply(all_curves$np.curves, predict, target_seq_depth)
  predicted_covarage = summary(predicted_coverage) * 100
  
  # Samples with highest and lowest coverage
  most_covered = which(coverage == max(coverage))
  least_covered = which(coverage == min(coverage))
  
  # Return variables as list
  nonpareil_results = list(curves = all_curves,
                           summary = result_all,
                           observed_effort = obs_seq_effort,
                           coverage_summary = coverage_percent, 
                           nearly_complete = nearly_complete,
                           coverage_prediction  = predicted_coverage,
                           highest_sample = most_covered,
                           lowest_sample = least_covered)
  
  return(nonpareil_results)
}


# nonpareil analyses
res = nonpareil_analyses(args[3:length(args)], args[1], args[2])
out_summary = paste0(args[2], '_summary.RDS')
saveRDS(res, file=out_summary)
write(paste0('File written: ', out_summary), stderr())
## writing individual tables
write_table = function(df, out_base, suffix){
    df = as.data.frame(df)
    df$Sample = rownames(df)
    df = cbind(df$Sample, df[,2:(ncol(df)-1)])
    colnames(df)[1] = 'Sample'
    F = paste(c(out_base, suffix), collapse='_')
    write.table(df, file=F, sep='\t', quote=FALSE, row.names=FALSE)
    write(paste0('File written: ', F), stderr())
}
#write_table(res$curve, args[2], 'curves.txt')
write_table(res$summary, args[2], 'summary.txt')
