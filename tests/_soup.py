"""A simulator with one ambient pool (soup) for all droplets, and its metrics.

tests/_data.py simulates the DecontX model itself: the contamination of a cell
comes from the other clusters only, in proportion to their native counts. Real
ambient RNA is one pool, released by all cell types, including the type of the
cell it lands in. This simulator can break each assumption of the model:

- own_soup: the soup includes the type of the cell. That share looks native, so
  the model cannot see it.
- lysis: some types release more RNA into the soup than their counts imply.
- soup_skew: some genes are more abundant in the soup than in any cell.
- subtypes: each cluster given to the method holds several true types.
- umi: the range of counts per cell.

With own_soup=False and no lysis, skew or subtypes it reproduces the model.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix


@dataclass
class SoupSim:
    """Simulated counts and their truth. Fractions are of the observed total."""

    X: csr_matrix  # observed counts, cells x genes
    z: np.ndarray  # cluster given to the method, 0-based
    cell_type: np.ndarray  # true type, 0-based; equals z without subtypes
    native: csr_matrix  # native counts
    contamination: np.ndarray  # soup counts / total
    foreign: np.ndarray  # soup counts from the types of other clusters / total
    soup_share: np.ndarray  # share of each cluster in the soup
    droplets: csr_matrix  # empty droplets drawn from the soup
    markers: list = field(default_factory=list)  # marker genes of each cluster
    phi: np.ndarray | None = None  # native profile of each type
    soup_profile: np.ndarray | None = None  # soup of the cells of each cluster


def simulate_soup(
    seed,
    sizes=(1500, 800, 400, 200, 100),
    n_genes=2000,
    delta=(4.0, 6.0),
    own_soup=True,
    lysis=None,
    soup_skew=0.0,
    subtypes=1,
    umi=(500, 1000),
    n_droplets=2000,
    droplet_umi=100,
    n_markers=20,
) -> SoupSim:
    """Simulate counts with one soup. sizes are the cells of each cluster.

    With subtypes > 1 each cluster holds that many types of equal size, whose
    profiles share half of their expression. lysis multiplies the soup share of
    each cluster (all its types). The soup share of a type is otherwise its
    expected native counts.
    """
    rng = np.random.default_rng(seed)
    n_clusters = len(sizes)
    n_types = n_clusters * subtypes
    cluster_of_type = np.arange(n_types) // subtypes
    phi = rng.dirichlet(np.full(n_genes, 0.1), size=n_clusters)[cluster_of_type]
    if subtypes > 1:
        phi = 0.5 * phi + 0.5 * rng.dirichlet(np.full(n_genes, 0.1), size=n_types)

    cell_type = np.concatenate(
        [
            np.repeat(k * subtypes + np.arange(subtypes), -(-s // subtypes))[:s]
            for k, s in enumerate(sizes)
        ]
    )
    z = cluster_of_type[cell_type]
    n_cells = len(z)

    total = rng.integers(umi[0], umi[1] + 1, size=n_cells)
    frac = rng.beta(*delta, size=n_cells)
    n_soup = rng.binomial(total, frac)

    lysis = np.ones(n_clusters) if lysis is None else np.asarray(lysis, float)
    weight = np.bincount(cell_type, weights=total - n_soup, minlength=n_types)
    weight = weight * lysis[cluster_of_type]
    skew = np.exp(soup_skew * rng.standard_normal(n_genes))
    soup_phi = phi * skew
    soup_phi /= soup_phi.sum(axis=1, keepdims=True)

    X = np.zeros((n_cells, n_genes))
    native = np.zeros((n_cells, n_genes))
    foreign = np.zeros(n_cells)
    for c in range(n_cells):
        native[c] = rng.multinomial(total[c] - n_soup[c], phi[cell_type[c]])
        w = weight.copy()
        if not own_soup:
            w[cluster_of_type == z[c]] = 0.0
        src = rng.multinomial(n_soup[c], w / w.sum())
        X[c] = native[c]
        for t in np.flatnonzero(src):
            X[c] += rng.multinomial(src[t], soup_phi[t])
        foreign[c] = src[cluster_of_type != z[c]].sum()

    pool = weight / weight.sum()
    soup = pool @ soup_phi
    soup_profile = np.empty((n_clusters, n_genes))
    for k in range(n_clusters):
        w = weight if own_soup else np.where(cluster_of_type == k, 0.0, weight)
        soup_profile[k] = (w / w.sum()) @ soup_phi
    droplets = rng.multinomial(droplet_umi, soup / soup.sum(), size=n_droplets)

    # Markers: the genes most specific to each cluster's mean profile.
    prof = np.stack([phi[cluster_of_type == k].mean(axis=0) for k in range(n_clusters)])
    spec = prof / (prof.sum(axis=0) - prof + 1e-12)
    markers = [np.argsort(-spec[k])[:n_markers] for k in range(n_clusters)]

    return SoupSim(
        X=csr_matrix(X),
        z=z,
        cell_type=cell_type,
        native=csr_matrix(native),
        contamination=n_soup / total,
        foreign=foreign / total,
        soup_share=np.bincount(cluster_of_type, weights=pool, minlength=n_clusters),
        droplets=csr_matrix(droplets),
        markers=markers,
        phi=phi,
        soup_profile=soup_profile,
    )


def oracle(sim: SoupSim):
    """The best per-count estimate: the posterior under the true parameters.

    Each count is native with probability theta phi / (theta phi + (1 - theta)
    soup), with the true type profile, soup and native fraction of its cell.
    It returns (contamination, counts), as a method would. Its metrics are the
    floor that the sampling noise leaves to any method.
    """
    X = sim.X.tocsr()
    rows = np.repeat(np.arange(X.shape[0]), np.diff(X.indptr))
    theta = 1 - sim.contamination[rows]
    p_native = theta * sim.phi[sim.cell_type[rows], X.indices]
    p_soup = (1 - theta) * sim.soup_profile[sim.z[rows], X.indices]
    counts = X.copy()
    counts.data = X.data * p_native / (p_native + p_soup)
    total = np.asarray(X.sum(axis=1)).ravel()
    return 1 - np.asarray(counts.sum(axis=1)).ravel() / total, counts


# Each scenario breaks one assumption of the model; "realistic" breaks them all.
# The base has five clusters of 1500 to 100 cells, 2000 genes and 500-1000 UMI.
SCENARIOS = {
    "S0_model": {"own_soup": False},
    "S1_own_soup": {},
    "S2_lysis": {"lysis": [1, 1, 1, 1, 10]},
    "S3_soup_skew": {"soup_skew": 1.0},
    "S4_subtypes": {"subtypes": 2},
    "S5_low_umi": {"umi": (200, 500)},
    "S6_realistic": {
        "lysis": [1, 1, 1, 1, 10],
        "soup_skew": 1.0,
        "subtypes": 2,
        "umi": (200, 1000),
    },
    # Two large clusters and a soup mildly unlike the cells (total variation
    # 0.12; 0.06 in the 10x human-mouse mixture, whose contamination also varies
    # in composition between cells). Run to convergence, the EM moves the
    # mismatch into the native profiles (tests/benchmarks/README.md, Section 3.8).
    "S7_drift": {
        "sizes": (1500, 1500),
        "n_genes": 3000,
        "umi": (2000, 12000),
        "soup_skew": 0.3,
    },
}


def score(sim: SoupSim, contamination, counts) -> dict:
    """Metrics of one estimate. counts are the decontaminated counts.

    marker_leak is the share of contaminating marker counts (markers of other
    clusters) that the decontaminated counts keep: 0 removes all, 1 none.
    over_removal is the share of native counts removed.
    """
    contamination = np.asarray(contamination, dtype=float)
    counts = csr_matrix(counts)
    N = sim.native
    total = np.asarray(sim.X.sum(axis=1)).ravel()
    removed = 1 - np.asarray(counts.sum(axis=1)).ravel() / total
    err = contamination - sim.contamination
    by_cluster = np.array([err[sim.z == k].mean() for k in range(len(sim.soup_share))])

    # Contaminating and kept counts of the markers of the other clusters.
    all_markers = np.concatenate(sim.markers)
    contam_counts = kept = 0.0
    for k, own in enumerate(sim.markers):
        rows = np.flatnonzero(sim.z == k)
        cols = np.setdiff1d(all_markers, own)
        Nk = N[rows][:, cols].toarray()
        contam_counts += (sim.X[rows][:, cols].toarray() - Nk).sum()
        kept += np.maximum(counts[rows][:, cols].toarray() - Nk, 0).sum()

    lost = N - counts
    lost.data = np.maximum(lost.data, 0)
    return {
        "mae_total": float(np.mean(np.abs(err))),
        "bias_total": float(np.mean(err)),
        "worst_cluster_bias": float(by_cluster[np.argmax(np.abs(by_cluster))]),
        "mae_foreign": float(np.mean(np.abs(contamination - sim.foreign))),
        "pearson_total": (
            float(np.corrcoef(contamination, sim.contamination)[0, 1])
            if contamination.std() > 0
            else float("nan")
        ),
        "counts_rel_error": float(
            np.sqrt(((counts - N).power(2)).sum() / N.power(2).sum())
        ),
        "marker_leak": float(kept.sum() / contam_counts) if contam_counts else 0.0,
        "over_removal": float(lost.sum() / N.sum()),
        "self_consistency": float(np.max(np.abs(contamination - removed))),
    }
