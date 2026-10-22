# Methods

## The DecontX model

DecontX {cite:p}`Yang2020` models the counts of each cell as a mixture of two parts:

1. **Native transcripts**, drawn from the expression profile `phi` of the cell's cluster.
2. **Contamination**, drawn from a profile `eta` that mixes the other clusters (or the empty droplets, when given).

An EM estimates, for each cell, the native proportion `theta` with a Beta prior `delta`, and from it the share of each count that is native.

## Same as celda

scDecontX implements celda 1.24.0's `decontX` step by step:

- The initial profiles are the `theta`-weighted counts of each cluster with a pseudocount of 1e-20, and `eta` of a cluster is the profile of all other clusters.
- The E-step gives each count the native probability `(phi + 1e-20)(theta + 1e-20)` against the contamination probability `(eta + 1e-20)(1 − theta + 1e-20)`.
- `delta` is refitted in every iteration with `MCMCprecision::fit_dirichlet` and its defaults: zeros shifted by 1% of the smallest positive value, a method-of-moments start, and a stop when no component changes by more than 0.1.
- `theta = (native counts + delta₁) / (total counts + delta₁ + delta₂)`.
- The EM stops when no `theta` changes by more than `tol` (celda's `convergence`, 1e-3), or after `max_iter` (500) iterations.
- The reported contamination is that of the last iteration's E-step; the corrected counts come from an E-step with the final parameters. As in celda, the two can differ by up to `tol`.
- With `background`, the pooled profile of the empty droplets replaces `eta`.

From the same initial `theta`, scDecontX and celda agree to within 6e-15 in every contamination estimate and to within 7e-13 in every corrected count (4e-14 on simulated data) (`tests/test_celda.py`, and Table 4 of `tests/benchmarks/results/tables.md`). The initial `theta` comes from R's random number generator, so a run with a given `random_state` draws different values, as two R seeds do.

## Why it is fast

celda 1.24.0 processes cells and their nonzero counts serially in C++ in each EM iteration. scDecontX stores one estimated native count for each nonzero of the CSR matrix and uses parallel Numba kernels for the main count operations:

- the E-step runs in parallel over cells;
- cluster accumulation runs in parallel over work units with similar numbers of nonzeros within each cluster, reducing load imbalance when cluster sizes are uneven;
- profile updates also traverse dense cluster × gene arrays;
- every sum has a fixed order, so the result does not depend on the number of threads.

On 20,000 cells × 15,000 genes a call takes 0.24 s, against 2.3 s for celda's EM and 29 s for a full `decontX()` call, which also computes a UMAP. `dtype="float32"` halves the memory of the counts and the output.

## Evidence

Numbers from `tests/benchmarks/`, all implementations at their defaults.

| | scDecontX | R celda | NiRuff | omicverse |
|---|---|---|---|---|
| Contamination MAE, 15 simulated datasets of 3,000 cells × 15,000 genes, 9–50% contamination | 0.088 | 0.088 | 0.099 | 0.088 |
| Max contamination difference from R, same start | 6e-15 | – | 0.061 | 9e-15 |
| Run time, 20,000 cells × 15,000 genes | **0.24 s** | 29 s (2.3 s EM only) | 1.7 s | 5.0 s |

On datasets used in the DecontX paper, scDecontX recovers the reported qualitative patterns. In the 10x human–mouse mixture, using species labels as clusters and no background, its estimated contamination closely matches each cell's other-species UMI fraction (RMSE 0.0002 in human and 0.0013 in mouse cells). This measures cross-species contamination, not total contamination, and the cell selection differs from the paper (12,165 cells here against 12,079). In the CellBench RNA mixtures, estimates correlate with the designed mixing proportions (Pearson r = 0.71 for CEL-seq2 and 0.86 for SORT-seq), although the median estimates differ from the published values. The mixing design does not measure contamination added during library preparation.

## Limits of the model

These hold for DecontX in general, and so for celda and scDecontX alike (`tests/benchmarks/README.md`, Sections 3.7–3.9):

- **Soup of the cell's own type looks native.** The contamination profile mixes the *other* clusters, so ambient RNA from the cell's own cell type is not counted. Without empty droplets the contamination of large clusters is underestimated; with them (`background`) the estimate is close to the truth.
- **Clusters should be broad cell types.** Splitting a cell type into subclusters puts the subtypes' shared expression into each other's contamination profile; in the human–mouse mixture four subclusters per species raised the RMSE against the cross-species truth from 0.0002–0.0013 to 0.03–0.06.
- **The stop rule matters.** Run to convergence (`tol` 1e-5), the EM slowly moves contamination into the native profiles whenever the real contamination profile differs from `eta`, as it does in real data. celda's `tol` of 1e-3 stops before that.

A leave-one-out variant, which removes each cell from its own cluster profile, was 3.7× more accurate on simulated data but not on real data with an external reference, so scDecontX keeps celda's estimator.

## Citing

Cite scDecontX together with the DecontX paper:

> Yang, S., Corbett, S. E., Koga, Y., Wang, Z., Johnson, W. E., Yajima, M., & Campbell, J. D. (2020). Decontamination of ambient RNA in single-cell RNA-seq with DecontX. *Genome Biology*, 21, 57. https://doi.org/10.1186/s13059-020-1950-6

> Siam, M. H. B. (2026). *scDecontX* (Version 1.0.0) [Computer software]. GitHub. https://github.com/mhbsiam/scdecontx

## Other methods

| Method | Ambient profile from | Notes |
|---|---|---|
| **SoupX** | empty droplets | estimates one contamination fraction per channel |
| **DecontX** | other clusters, or empty droplets | estimates a fraction per cell |
| **CellBender** | a deep generative model of the raw matrix | needs the unfiltered matrix and a GPU for speed |
