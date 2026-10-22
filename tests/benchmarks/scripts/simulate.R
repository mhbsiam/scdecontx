# Simulate contaminated counts with celda::simulateContamination.
# Usage: Rscript simulate.R <out_dir> <C> <G> <K> <seed> <delta1> <delta2>

args <- commandArgs(trailingOnly = TRUE)
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))), "io.R"))
suppressPackageStartupMessages(library(celda))

out <- args[1]
C <- as.integer(args[2])
G <- as.integer(args[3])
K <- as.integer(args[4])
seed <- as.integer(args[5])
delta <- c(as.numeric(args[6]), as.numeric(args[7]))

sim <- simulateContamination(C = C, G = G, K = K, delta = delta, seed = seed)

write_csc(Matrix::Matrix(sim$observedCounts, sparse = TRUE), out, "counts")
write_csc(Matrix::Matrix(sim$nativeCounts, sparse = TRUE), out, "native")
write_vec(sim$z, file.path(out, "z.bin"))
write_vec(sim$contamination, file.path(out, "true_contamination.bin"))
