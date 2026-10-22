# Run celda::decontX on a dataset and save the results.
# Usage: Rscript run_r.R <data_dir> <out_dir> <max_iter> <convergence> <seed> <repeats>
#
# trace() captures the initial theta so that the Python implementations can
# start from the same values. The script times the UMAP that celda always
# runs separately.

args <- commandArgs(trailingOnly = TRUE)
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))), "io.R"))
suppressPackageStartupMessages(library(celda))

data_dir <- args[1]
out <- args[2]
max_iter <- as.integer(args[3])
convergence <- as.numeric(args[4])
seed <- as.integer(args[5])
repeats <- as.integer(args[6])
dir.create(out, showWarnings = FALSE, recursive = TRUE)

counts <- read_csc(data_dir, "counts")
z <- readBin(file.path(data_dir, "z.bin"), "double", n = ncol(counts), size = 8)
rownames(counts) <- paste0("g", seq_len(nrow(counts)))
colnames(counts) <- paste0("c", seq_len(ncol(counts)))

ns <- asNamespace("celda")
invisible(utils::capture.output(suppressMessages({
  trace("decontXInitialize", where = ns, print = FALSE,
    tracer = quote(assign("THETA0", theta, envir = .GlobalEnv)))
  trace(".decontxInitializeZ", where = ns, print = FALSE,
    tracer = quote(assign("UMAP_T0", proc.time()[["elapsed"]], envir = .GlobalEnv)),
    exit = quote(assign("UMAP_T1", proc.time()[["elapsed"]], envir = .GlobalEnv)))
})))

totals <- c()
umaps <- c()
for (r in seq_len(repeats)) {
  t0 <- proc.time()[["elapsed"]]
  res <- decontX(counts, z = z, maxIter = max_iter, convergence = convergence,
                 delta = c(10, 10), estimateDelta = TRUE, seed = seed, verbose = FALSE)
  totals <- c(totals, proc.time()[["elapsed"]] - t0)
  umaps <- c(umaps, UMAP_T1 - UMAP_T0)
}

est <- res$estimates[[1]]
write_vec(res$contamination, file.path(out, "contamination.bin"))
write_vec(est$theta, file.path(out, "theta.bin"))
write_vec(THETA0, file.path(out, "theta0.bin"))
write_csc(res$decontXcounts, out, "decontx")
writeLines(jsonlite::toJSON(list(
  total_s = totals,
  umap_s = umaps,
  em_s = totals - umaps,
  n_iter = est$iteration,
  delta = as.numeric(est$delta),
  celda_version = as.character(packageVersion("celda")),
  r_version = R.version.string
), auto_unbox = TRUE, digits = NA), file.path(out, "meta.json"))
