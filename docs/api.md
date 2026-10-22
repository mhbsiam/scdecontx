# API

`scdecontx` has one function.

```{eval-rst}
.. module:: scdecontx

.. autosummary::
    :toctree: generated

    run
```

```python
import scdecontx as dx

dx.run(adata, "leiden")
```

## Agreement with R

`run` is celda's `decontX`: the same model, prior fit (`MCMCprecision::fit_dirichlet`), defaults (`delta = (10, 10)`, `tol` 1e-3, `max_iter` 500) and reported contamination (that of the last iteration). From the same initial values the contamination estimates agree to within 6e-15 and the corrected counts to within 7e-13. The initial values come from R's random number generator, so a run with `random_state` draws different ones, as two R seeds do. See [Methods](methods.md).

## Output

`obs[f"{key_added}_contamination"]`
: Contamination fraction of each cell. `NaN` for cells without counts.

`layers[f"{key_added}_counts"]`
: Decontaminated counts. Fractional values with the sparsity pattern of the input.

`uns[key_added]`
: - `params`: the arguments except `copy` and `key_added`, with the `tol` and `max_iter` that were used. `background` is `True` or `False`, not the droplets.
  - `fit`: parallel lists with one entry for each batch: `batch`, `delta_native`, `delta_contamination`, `n_iter`, `converged`, `final_change`.
  - `background`: `n_droplets` (droplets used) and `n_dropped_overlap` (droplets that were also cells).
  - `version`.

## Convergence

A batch that reaches `max_iter` before no native proportion changes by `tol` gives a `UserWarning`, also when logging is off, and its `fit["converged"]` is `False`. Raise `max_iter` or `tol` and run again. Progress goes to the `scdecontx` logger:

```python
import logging

logging.getLogger("scdecontx").setLevel(logging.INFO)
```
