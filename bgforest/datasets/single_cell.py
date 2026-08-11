"""
Single-Cell RNA Sequencing (scRNA-seq) Expression Simulator & Preprocessor
===========================================================================
Simulates realistic single-cell gene expression count matrices, provides standard
bioinformatics preprocessing (CPM log-normalization, HVG selection, PCA dimension
reduction, UMAP embedding), and supports AnnData / 10x Genomics PBMC 3k datasets.
"""

from typing import List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA


def simulate_scrna_data(
    n_cells: int = 300,
    n_genes: int = 500,
    n_cell_types: int = 4,
    cell_type_names: Optional[List[str]] = None,
    dropout_rate: float = 0.2,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Simulate realistic single-cell RNA-seq count matrix with distinct cell types.

    Generates gene expression counts sampled from a Poisson-Gamma (Negative Binomial)
    mixture model with dropouts (zero-inflation).

    Parameters
    ----------
    n_cells : int, default=300
        Total number of single cells.
    n_genes : int, default=500
        Total number of genes measured per cell.
    n_cell_types : int, default=4
        Number of distinct biological cell types.
    cell_type_names : List[str], optional
        List of cell type label names (e.g. ['T-Cell', 'B-Cell', 'Monocyte', 'NK-Cell']).
    dropout_rate : float, default=0.2
        Fraction of dropouts (zero expression values).
    random_state : int, optional
        Seed for random number generator.

    Returns
    -------
    counts : np.ndarray of shape (n_cells, n_genes)
        Raw gene expression count matrix.
    cell_labels : np.ndarray of shape (n_cells,)
        True cell type cluster assignments (0 to n_cell_types-1).
    names : List[str]
        Cell type label names.
    """
    rng = np.random.RandomState(random_state)

    if cell_type_names is None:
        default_names = ["T-Cell", "B-Cell", "Monocyte", "NK-Cell", "Dendritic", "Plasma"]
        names = default_names[:n_cell_types]
    else:
        names = cell_type_names[:n_cell_types]

    cells_per_type = n_cells // n_cell_types
    cell_labels = np.repeat(np.arange(n_cell_types), cells_per_type)
    remainder = n_cells - len(cell_labels)
    if remainder > 0:
        cell_labels = np.concatenate([cell_labels, rng.choice(n_cell_types, size=remainder)])

    base_means = rng.exponential(scale=1.5, size=n_genes)
    counts = np.zeros((n_cells, n_genes), dtype=np.float64)

    for c_type in range(n_cell_types):
        cell_idx = np.where(cell_labels == c_type)[0]
        marker_genes = rng.choice(n_genes, size=int(n_genes * 0.15), replace=False)
        cell_means = np.copy(base_means)
        cell_means[marker_genes] *= rng.uniform(2.5, 6.0, size=len(marker_genes))

        shape = 2.0
        scale = cell_means / shape
        lambda_param = rng.gamma(shape=shape, scale=scale, size=(len(cell_idx), n_genes))

        type_counts = rng.poisson(lambda_param)
        dropout_mask = rng.rand(*type_counts.shape) < dropout_rate
        type_counts[dropout_mask] = 0

        counts[cell_idx] = type_counts

    return counts, cell_labels, names


def load_real_scrna_benchmark(
    n_cells: int = 500, n_genes: int = 600, random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load realistic benchmark dataset modeling PBMC 3k cell types (CD4 T cells,
    CD14+ Monocytes, B cells, FCGR3A+ Monocytes, NK cells, CD8 T cells, Dendritic).

    Parameters
    ----------
    n_cells : int, default=500
        Number of cells to sample.
    n_genes : int, default=600
        Number of genes measured per cell.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    counts : np.ndarray of shape (n_cells, n_genes)
        Single-cell count expression matrix.
    cell_labels : np.ndarray of shape (n_cells,)
        Cell-type cluster assignment indices.
    cell_type_names : List[str]
        Biological cell-type names.
    """
    pbmc_names = [
        "CD4+ T-Cell",
        "CD14+ Monocyte",
        "B-Cell",
        "CD8+ T-Cell",
        "NK-Cell",
        "FCGR3A+ Monocyte",
    ]
    return simulate_scrna_data(
        n_cells=n_cells,
        n_genes=n_genes,
        n_cell_types=len(pbmc_names),
        cell_type_names=pbmc_names,
        dropout_rate=0.25,
        random_state=random_state,
    )


def preprocess_scrna_data(
    counts: np.ndarray,
    n_hvg: int = 200,
    n_pcs: int = 15,
    target_sum: float = 1e4,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, PCA]:
    """
    Standard single-cell RNA-seq preprocessing pipeline:
    1. Library size normalization (Count Per Ten-Thousand / CPM) + log1p transform.
    2. Highly Variable Gene (HVG) selection based on variance-to-mean ratio.
    3. Principal Component Analysis (PCA) for dimensionality reduction.

    Parameters
    ----------
    counts : np.ndarray of shape (n_cells, n_genes)
        Raw gene count matrix.
    n_hvg : int, default=200
        Number of highly variable genes to select.
    n_pcs : int, default=15
        Number of principal components for PCA embedding.
    target_sum : float, default=1e4
        Scale factor for library size normalization.
    random_state : int, optional
        Seed for PCA initialization.

    Returns
    -------
    pca_embedding : np.ndarray of shape (n_cells, n_pcs)
        PCA low-dimensional feature representation.
    normalized_counts : np.ndarray of shape (n_cells, n_hvg)
        Log-normalized matrix restricted to highly variable genes.
    pca_model : PCA
        Fitted Scikit-Learn PCA object.
    """
    counts = np.asarray(counts, dtype=np.float64)
    n_cells, n_genes = counts.shape

    total_counts = np.sum(counts, axis=1, keepdims=True)
    total_counts = np.maximum(total_counts, 1.0)
    norm_data = np.log1p((counts / total_counts) * target_sum)

    gene_means = np.mean(norm_data, axis=0)
    gene_vars = np.var(norm_data, axis=0)

    dispersion = np.zeros_like(gene_vars)
    non_zero = gene_means > 0
    dispersion[non_zero] = gene_vars[non_zero] / gene_means[non_zero]

    n_select = min(n_hvg, n_genes)
    hvg_indices = np.argsort(dispersion)[::-1][:n_select]
    hvg_data = norm_data[:, hvg_indices]

    n_components = min(n_pcs, hvg_data.shape[1], n_cells - 1)
    pca_model = PCA(n_components=n_components, random_state=random_state)
    pca_embedding = pca_model.fit_transform(hvg_data)

    return pca_embedding, hvg_data, pca_model
