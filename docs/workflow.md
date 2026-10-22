# Workflow

## Where DecontX fits

DecontX models counts. It needs two things:

- raw counts in `adata.X` (or in a layer)
- cluster labels in `adata.obs`

```
load  →  QC filter  →  cluster (normalized copy)  →  dx.run (raw counts)  →  normalize → analyze
```

Clustering needs normalized data, and DecontX needs raw counts, so cluster on a copy and keep `adata.X` raw.

```python
import scanpy as sc
import scdecontx as dx

adata = sc.read_h5ad("pbmc.h5ad")  # adata.X holds raw counts

# 1. Quality filters work on counts.
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# 2. Cluster a normalized copy. adata.X stays raw.
clust = adata.copy()
sc.pp.normalize_total(clust)
sc.pp.log1p(clust)
sc.pp.highly_variable_genes(clust, n_top_genes=2000)
sc.pp.pca(clust)
sc.pp.neighbors(clust)
sc.tl.leiden(clust)
adata.obs["leiden"] = clust.obs["leiden"]
del clust

# 3. Remove the contamination.
dx.run(adata, "leiden")
print(f"Mean contamination: {adata.obs['scdecontx_contamination'].mean():.1%}")

# 4. Continue from the decontaminated counts.
adata.layers["raw_counts"] = adata.X.copy()
adata.X = adata.layers["scdecontx_counts"]
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
```

## Existing labels and layers

Any `obs` column works as `groupby`, including annotations. Counts in a layer are read from there; `.X` is not touched.

```python
dx.run(adata, "cell_type")
dx.run(adata, "leiden", layer="counts")
```

## Batches

Samples with their own ambient RNA should be fitted separately:

```python
dx.run(adata, "leiden", batch_key="sample")
```

## Empty droplets

The raw (unfiltered) matrix contains empty droplets, which measure the ambient profile directly. Pass them as `background`. DecontX then fixes the contamination profile to the droplet profile instead of estimating it from the other clusters.

```python
raw = sc.read_10x_h5("raw_feature_bc_matrix.h5")
raw.var_names_make_unique()
empty = raw[raw.X.sum(axis=1).A1 < 100]  # droplets, not cells

dx.run(adata, "leiden", background=empty)
```

Barcodes that are also cells in `adata` are removed from the background. With batches, label the droplets with the same batch names. Droplets of a batch without cells are removed with a warning:

```python
dx.run(adata, "leiden", batch_key="sample", background=empty, background_batch_key="sample")
```

## Common errors

- **Running after `sc.pp.log1p`.** This raises `ValueError`. Run DecontX first, or pass the raw layer.
- **Running after `sc.pp.normalize_total` only.** This gives a warning about non-integer values, and the results are unreliable.
- **Subsetting to highly variable genes first.** DecontX estimates the ambient profile from all genes. Run it on all genes, then subset.

Filter empty barcodes and never-detected genes before DecontX. A cell with zero total counts gets `NaN` contamination and a warning.

## The output

- `adata.obs["scdecontx_contamination"]`: the contamination fraction of each cell.
- `adata.layers["scdecontx_counts"]`: the decontaminated counts, with the sparsity of the input. They are fractional, as in R. Round them if a tool needs integers; rounding zeroes every count below 0.5.
- `adata.uns["scdecontx"]`: the parameters, the fitted prior `delta` and the convergence of each batch.

```python
removed = adata.layers["raw_counts"] - adata.layers["scdecontx_counts"]
```
