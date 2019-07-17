#!/usr/bin/env Rscript

# libraries
suppressPackageStartupMessages(library("argparse"))
suppressPackageStartupMessages(library("data.table"))

# create parser object
parser <- ArgumentParser()

# specifying options
parser$add_argument("tableX", nargs=1, help="First table to join on (tab-delim with header)")
parser$add_argument("tableY", nargs=1, help="Second table to join on (tab-delim with header)")
parser$add_argument("-x", "--columnX", type='character', default='V1',
			   help="Columns to join on (comma-delim) for Table1 [default: %(default)s]")
parser$add_argument("-y", "--columnY", type='character', default='V1',
			   help="Columns to join on (comma-delim) for Table2 [default: %(default)s]")
parser$add_argument("-X", "--allX", action="store_true", default=TRUE,
			   help="Keep all x-table rows? [default: %(default)s]")
parser$add_argument("-Y", "--allY", action="store_true", default=TRUE,
			   help="Keep all y-table rows? [default: %(default)s]")
parser$add_argument("-o", "--output", type='character', default='merged.tsv',
			   help="Path for output [default: %(default)s]")
parser$add_argument("-v", "--verbose", action="store_true", default=TRUE,
			   help="Print extra output [default: %(default)s]")
parser$add_argument("-q", "--quietly", action="store_false",
			   dest="verbose", help="Print little output")
args <- parser$parse_args()


# loading tables and joining
fwrite(
  merge(
    fread(args[['tableX']], sep='\t'),
    fread(args[['tableY']], sep='\t'),
    by.x = unlist(strsplit(args[['columnX']], ',')),
    by.y = unlist(strsplit(args[['columnY']], ',')),
    all.x = args[['allX']],
    all.y = args[['allY']]
    ),
  file=args[['output']],
  sep='\t',
  quote=FALSE,
  row.names=FALSE
  )
write(sprintf('File written:: %s', args[['output']]), stderr())