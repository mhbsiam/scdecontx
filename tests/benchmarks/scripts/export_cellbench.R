# Export the CellBench SingleCellExperiments to binary CSC and CSV for Python.
# Usage: Rscript export_cellbench.R <dir with the .RData files>
# Writes <dir>/export/<object>/{counts_*.bin, counts_dim.txt, genes.txt, cells.csv}.

args <- commandArgs(trailingOnly = TRUE)
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))), "io.R"))
suppressPackageStartupMessages(library(SingleCellExperiment))

dir <- args[1]
for (f in c("mRNAmix_qc.RData", "sincell_with_class.RData", "sincell_with_class_5cl.RData")) {
  e <- new.env()
  for (o in load(file.path(dir, f), envir = e)) {
    sce <- get(o, envir = e)
    out <- file.path(dir, "export", o)
    write_csc(counts(sce), out, "counts")
    writeLines(rownames(sce), file.path(out, "genes.txt"))
    meta <- as.data.frame(colData(sce))
    meta$barcode <- colnames(sce)
    write.csv(meta, file.path(out, "cells.csv"), row.names = FALSE)
    cat(o, ncol(sce), "cells x", nrow(sce), "genes\n")
  }
}
