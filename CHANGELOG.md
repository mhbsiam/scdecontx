# Changelog

## 1.0.0

First release of scDecontX: celda's DecontX for scverse, with the same results and 6–10× the speed.

### Method

scDecontX is celda 1.24.0's `decontX`: the same model, initial profiles, `MCMCprecision::fit_dirichlet` prior fit, defaults (`delta = (10, 10)`, `tol` 1e-3, `max_iter` 500) and reported contamination (that of the last iteration). From the same initial values it agrees with R to within 6e-15 in every contamination estimate and to within 7e-13 in every corrected count.

A leave-one-out variant, which removes each cell from its own cluster profile, was 3.7× more accurate on simulated data but not on real data with an external reference (a human–mouse mixture, the CellBench RNA mixtures, pbmc4k against sorted PBMCs): run to convergence, it moved contamination into the native profiles. It is not part of the release; `tests/benchmarks/README.md` documents the comparison.

### API

`dx.run(adata, groupby, ...)` is the whole public API: raw counts from `.X` or a layer, clusters and optional batches from `obs`, empty droplets as `background`, results in `obs["scdecontx_contamination"]`, `layers["scdecontx_counts"]` and `uns["scdecontx"]`. Cells with zero counts get `NaN` contamination and do not influence the fit; log-normalized input is rejected.

### Performance

Sparse Numba kernels over the CSR nonzeros, parallel over cells and over runs of cells within clusters, with a fixed summation order, so results do not depend on the thread count and one dominant cluster does not leave cores idle. On 20,000 cells × 15,000 genes a call takes 0.24 s, against 2.3 s for celda's EM and 29 s for a full R `decontX()` call. `dtype="float32"` halves memory.

### Testing and benchmark

Tests compare scDecontX with celda fixtures, the kernels with a NumPy oracle, and the estimates with simulated truth, including a simulator in which all droplets share one ambient pool. `tests/benchmarks/` compares scDecontX with R celda, NiRuff/decontx-python and omicverse/py-decontx, and checks it on the datasets of the DecontX paper; see `tests/benchmarks/README.md`.

## Acknowledgments

- DecontX is by Yang et al. (2020), *Genome Biology* 21:57, and its reference implementation is in the R package [celda](https://github.com/campbio/celda) (MIT). scDecontX follows celda's `decontX`.
- The CSR-aligned EM, which keeps one native count for each nonzero and sums them into cluster profiles in Numba, follows a design by Jason Jia ([jjia1/decontx-python](https://github.com/jjia1/decontx-python)).
- Nicolas Ruffini's [decontx-python](https://github.com/NiRuff/decontx-python) showed how DecontX fits into an AnnData workflow, and is one of the implementations the benchmark compares against.
