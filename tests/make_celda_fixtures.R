# Make the celda reference fixtures for tests/test_celda.py.
# Usage: Rscript tests/make_celda_fixtures.R
#
# For each dataset, write the counts, the cluster labels, the initial theta
# that celda draws (captured with trace()) and celda's results to
# tests/fixtures/celda/<name>/. scdecontx started from the same
# theta must reproduce the results.

args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("--file=", "", grep("--file=", args, value = TRUE))))
source(file.path(here, "benchmarks", "scripts", "io.R"))
suppressPackageStartupMessages(library(celda))
out_root <- file.path(here, "fixtures", "celda")

invisible(utils::capture.output(suppressMessages(
  trace("decontXInitialize", where = asNamespace("celda"), print = FALSE,
    tracer = quote(assign("THETA0", theta, envir = .GlobalEnv)))
)))

make <- function(name, C, G, K, delta, seed, background = FALSE) {
  sim <- simulateContamination(C = C, G = G, K = K, delta = delta, seed = seed)
  counts <- Matrix::Matrix(sim$observedCounts, sparse = TRUE)
  rownames(counts) <- paste0("g", seq_len(G))
  colnames(counts) <- paste0("c", seq_len(C))
  bg <- NULL
  if (background) {
    # Empty droplets drawn from the pooled expression of all cells.
    set.seed(seed)
    profile <- Matrix::rowSums(counts) / sum(counts)
    bg <- sapply(seq_len(100), function(i) rmultinom(1, 30, profile))
    bg <- Matrix::Matrix(bg, sparse = TRUE)
    rownames(bg) <- rownames(counts)
    colnames(bg) <- paste0("e", seq_len(100))
  }
  res <- decontX(counts, z = sim$z, background = bg, maxIter = 500,
                 convergence = 0.001, delta = c(10, 10), estimateDelta = TRUE,
                 seed = 12345, verbose = FALSE)
  est <- res$estimates[[1]]
  dir <- file.path(out_root, name)
  write_csc(counts, dir, "counts")
  if (!is.null(bg)) write_csc(bg, dir, "background")
  write_vec(sim$z, file.path(dir, "z.bin"))
  write_vec(THETA0, file.path(dir, "theta0.bin"))
  write_vec(res$contamination, file.path(dir, "contamination.bin"))
  write_vec(est$theta, file.path(dir, "theta.bin"))
  write_vec(est$delta, file.path(dir, "delta.bin"))
  write_csc(res$decontXcounts, dir, "decontx")
  writeLines(as.character(est$iteration), file.path(dir, "n_iter.txt"))
}

make("low", C = 200, G = 150, K = 3, delta = c(1, 10), seed = 1)
make("high", C = 200, G = 150, K = 4, delta = c(4, 6), seed = 2)
make("background", C = 200, G = 150, K = 3, delta = c(2, 8), seed = 3, background = TRUE)
writeLines(paste("celda", packageVersion("celda")), file.path(out_root, "VERSION"))
