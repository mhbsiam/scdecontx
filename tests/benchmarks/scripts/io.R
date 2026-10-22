# Read and write a dgCMatrix (genes x cells) as raw binary CSC slots.
# The same bytes are a cells x genes CSR matrix in Python.

write_csc <- function(m, dir, prefix) {
  m <- methods::as(m, "CsparseMatrix")
  m <- methods::as(m, "generalMatrix")
  m <- methods::as(m, "dMatrix")
  dir.create(dir, showWarnings = FALSE, recursive = TRUE)
  writeLines(as.character(dim(m)), file.path(dir, paste0(prefix, "_dim.txt")))
  writeBin(as.integer(m@i), file.path(dir, paste0(prefix, "_i.bin")), size = 4)
  writeBin(as.integer(m@p), file.path(dir, paste0(prefix, "_p.bin")), size = 4)
  writeBin(as.double(m@x), file.path(dir, paste0(prefix, "_x.bin")), size = 8)
}

read_csc <- function(dir, prefix) {
  d <- as.integer(readLines(file.path(dir, paste0(prefix, "_dim.txt"))))
  p <- readBin(file.path(dir, paste0(prefix, "_p.bin")), "integer", n = d[2] + 1, size = 4)
  nnz <- p[length(p)]
  i <- readBin(file.path(dir, paste0(prefix, "_i.bin")), "integer", n = nnz, size = 4)
  x <- readBin(file.path(dir, paste0(prefix, "_x.bin")), "double", n = nnz, size = 8)
  Matrix::sparseMatrix(i = i, p = p, x = x, dims = d, index1 = FALSE)
}

write_vec <- function(v, path) writeBin(as.double(v), path, size = 8)
