# Performance

## Threads

The main count operations run in parallel Numba kernels using the available Numba threads. Cluster accumulation is split into work units with similar numbers of nonzeros, reducing load imbalance when cluster sizes are uneven. Runtime still depends on the data and cluster structure. The result does not depend on the number of threads. To limit the threads, set `NUMBA_NUM_THREADS` before Python starts.

## The first call

Numba compiles the kernels on the first call and caches them on disk. Later processes load the cache.

## Speed

scDecontX runs celda's algorithm with celda's stop rule, 6–10× faster than celda's EM and about 120× faster than a full R `decontX()` call, which also computes a UMAP: on 20,000 cells × 15,000 genes 0.24 s against 2.3 s and 29 s (`tests/benchmarks/README.md`).

## Memory

The counts, the native counts and the output layer each take one value for each nonzero. `dtype="float32"` halves all three. The other arrays are small: one value for each cell and two for each cluster and gene.

```python
dx.run(adata, "leiden", dtype="float32")
```

## Batches

Each batch is fitted separately. Many small batches cost little more than one large one.
