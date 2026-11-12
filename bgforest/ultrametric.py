"""
Ultrametrics around spanning trees
==================================

Three standard facts, each checked against enumeration in ``scripts/verify_exactness.py``.

* **Single linkage is the best ultrametric below the data.** The minimax path
  distance u(i, j) (the largest edge on the minimum-spanning-tree path) is the
  largest ultrametric that never exceeds the input distance (Gower & Ross, 1969).
* **Kirchhoff melts into single linkage.** With conductances exp(-lam c_e),
  (1/lam) log R_eff converges to the minimax path cost uniformly, with error at
  most 2 log(n) / lam.
* **Brownian motion on a tree has an ultrametric covariance.** Diffusing from a
  root along a tree, the covariance of two nodes is the length of the common part
  of their root paths; its precision is the grounded tree Laplacian.
"""

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from bgforest.treelaw import TreeLaw, mst_edges


def single_linkage_ultrametric(D: np.ndarray) -> np.ndarray:
    """Minimax path distance: u(i, j) = the largest edge on the MST path from i to j.

    Equivalently the cophenetic distance of single-linkage clustering, and the
    subdominant ultrametric of D (the largest ultrametric with u <= D).
    """
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    U = np.zeros((n, n))
    if n <= 1:
        return U
    adj = [[] for _ in range(n)]
    for i, j in mst_edges(D):
        adj[i].append((j, D[i, j]))
        adj[j].append((i, D[i, j]))
    for s in range(n):
        best = np.full(n, -np.inf)
        best[s] = 0.0
        stack = [s]
        seen = np.zeros(n, dtype=bool)
        seen[s] = True
        while stack:
            u = stack.pop()
            for v, w in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    best[v] = max(best[u], w)
                    stack.append(v)
        U[s] = best
    U = np.maximum(U, U.T)
    np.fill_diagonal(U, 0.0)
    return U


def is_ultrametric(U: np.ndarray, tol: float = 1e-12) -> bool:
    """True if U is symmetric with zero diagonal and U(i, j) <= max(U(i, k), U(k, j))."""
    U = np.asarray(U, dtype=np.float64)
    if not np.allclose(U, U.T, atol=tol) or np.abs(np.diag(U)).max(initial=0.0) > tol:
        return False
    # bound[i, j] = min_k max(U[i, k], U[k, j]) never exceeds U[i, j] (take k = i);
    # U is ultrametric exactly when no two-step path is shorter in the max sense.
    bound = U.copy()
    for k in range(U.shape[0]):
        np.minimum(bound, np.maximum(U[:, k][:, None], U[k, :][None, :]), out=bound)
    return bool(np.all(U <= bound + tol))


def log_resistance_ultrametric(C: np.ndarray, lam: float) -> np.ndarray:
    """(1/lam) log R_eff for conductances exp(-lam C): tends to the minimax path cost of C.

    ``|(1/lam) log R_eff - u| <= 2 log(n) / lam`` for every pair, where u is
    :func:`single_linkage_ultrametric` of C.
    """
    law = TreeLaw.from_costs(np.asarray(C, dtype=np.float64), lam)
    out = law.log_resistance() / float(lam)
    np.fill_diagonal(out, 0.0)
    return out


def resistance_bound(n: int, lam: float) -> float:
    """The uniform error bound 2 log(n) / lam of :func:`log_resistance_ultrametric`."""
    return 2.0 * np.log(n) / float(lam)


def _children(parent: np.ndarray):
    kids = [[] for _ in range(len(parent))]
    for v, p in enumerate(parent):
        if p >= 0:
            kids[int(p)].append(v)
    return kids


def reroot(edges: Iterable[Tuple[int, int]], n: int, root: int) -> np.ndarray:
    """Parent array of the tree with the given undirected edges, rooted at ``root``."""
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    parent = np.full(n, -2, dtype=np.int64)
    parent[root] = -1
    stack = [root]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if parent[v] == -2:
                parent[v] = u
                stack.append(v)
    if (parent == -2).any():
        raise ValueError("the edges do not span a tree on n nodes.")
    return parent


def tree_covariance(parent: np.ndarray, edge_variance: Optional[Sequence[float]] = None) -> np.ndarray:
    """Covariance of Brownian motion run from the root along a tree.

    ``G[i, j]`` is the total variance of the edges shared by the root paths of i and
    j (with unit variances: the depth of their meet). G restricted to the non-root
    nodes is the inverse of the grounded tree Laplacian, its determinant is 1 for
    unit variances, and it is an ultrametric matrix: G[i, j] >= min(G[i, k], G[k, j]).

    Parameters
    ----------
    parent : (n,) int array
        Parent pointers, -1 at the root.
    edge_variance : (n,) array, optional
        Variance of the step from ``parent[v]`` to v, indexed by the child v. Defaults to 1.
    """
    parent = np.asarray(parent, dtype=np.int64)
    n = len(parent)
    var = np.ones(n) if edge_variance is None else np.asarray(edge_variance, dtype=np.float64)
    roots = np.flatnonzero(parent < 0)
    if len(roots) != 1:
        raise ValueError("parent must describe one rooted tree.")
    # A[v, e] = 1 if the edge above node e lies on v's root path; then G = A diag(var) A'.
    A = np.zeros((n, n))
    order = [int(roots[0])]
    kids = _children(parent)
    for u in order:
        order.extend(kids[u])
    if len(order) != n:
        raise ValueError("parent contains a cycle.")
    for v in order[1:]:
        A[v] = A[parent[v]]
        A[v, v] = 1.0
    return (A * var) @ A.T


def clade_haar_basis(parent: np.ndarray) -> Tuple[np.ndarray, np.ndarray, list]:
    """Orthonormal Haar basis of the clade hierarchy of a rooted tree.

    The hierarchy holds the clade D(v) (v and its descendants) of every node, plus the
    singleton {v} of every internal node. The children of D(v) are {v} and the clades
    of v's children. Each hierarchy node with m children contributes m - 1 vectors that
    are constant on its children and sum to zero on it; the constant vector completes
    the basis.

    Any matrix of the form sum_B alpha_B 1_B 1_B' over this hierarchy, such as
    :func:`tree_covariance`, has W' A W supported on pairs of basis vectors whose
    hierarchy nodes are nested (equal, or one an ancestor of the other).

    Returns
    -------
    W : (n, n) array
        Orthonormal columns; column 0 is the constant vector.
    owner : (n,) int array
        For column k >= 1, the tree node v whose clade D(v) the vector belongs to; -1 for
        the constant.
    clades : list of sets
        ``clades[v]`` is D(v).
    """
    parent = np.asarray(parent, dtype=np.int64)
    n = len(parent)
    kids = _children(parent)
    root = int(np.flatnonzero(parent < 0)[0])
    order = [root]
    for u in order:
        order.extend(kids[u])
    clades = [set() for _ in range(n)]
    for v in reversed(order):
        clades[v] = {v}.union(*[clades[c] for c in kids[v]])
    W = np.zeros((n, n))
    owner = np.full(n, -1, dtype=np.int64)
    W[:, 0] = 1 / np.sqrt(n)
    col = 1
    for v in order:
        parts = [[v]] + [sorted(clades[c]) for c in kids[v]]
        merged = list(parts[0])
        for part in parts[1:]:
            a, b = len(merged), len(part)
            vec = np.zeros(n)
            vec[merged] = -b
            vec[part] = a
            W[:, col] = vec / np.linalg.norm(vec)
            owner[col] = v
            col += 1
            merged += part
    return W, owner, clades


def tree_diffusion(
    n: int,
    sigma: float = 1.0,
    dim: int = 2,
    parent: Optional[np.ndarray] = None,
    random_state=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Grow a point cloud by Gaussian diffusion along a tree.

    A tree is drawn uniformly from all n^(n-2) labelled trees (unless ``parent`` is
    given) and rooted at node 0; every step from parent to child adds an independent
    N(0, sigma^2 I) displacement. Given the points, the posterior over trees under a
    flat prior is :class:`bgforest.treelaw.TreeLaw` with ``lam = 1 / (2 sigma^2)``.

    Returns
    -------
    X : (n, dim) array
    parent : (n,) int array, -1 at the root
    """
    rng = np.random.default_rng(random_state)
    if parent is None:
        if n <= 2:
            edges = [(0, 1)] if n == 2 else []
        else:
            edges = _prufer_tree(rng.integers(0, n, n - 2), n)
        parent = reroot(edges, n, 0)
    parent = np.asarray(parent, dtype=np.int64)
    X = np.zeros((n, dim))
    order = [int(np.flatnonzero(parent < 0)[0])]
    kids = _children(parent)
    for u in order:
        order.extend(kids[u])
    steps = rng.normal(0.0, sigma, (n, dim))
    for v in order[1:]:
        X[v] = X[parent[v]] + steps[v]
    return X, parent


def _prufer_tree(seq: Sequence[int], n: int):
    """Edges of the labelled tree with Pruefer code ``seq`` (linear-time decoder)."""
    import heapq

    degree = np.ones(n, dtype=np.int64)
    for x in seq:
        degree[x] += 1
    leaves = [i for i in range(n) if degree[i] == 1]
    heapq.heapify(leaves)
    edges = []
    for x in seq:
        leaf = heapq.heappop(leaves)
        edges.append((leaf, int(x)))
        degree[x] -= 1
        if degree[x] == 1:
            heapq.heappush(leaves, int(x))
    u, v = heapq.heappop(leaves), heapq.heappop(leaves)
    edges.append((u, v))
    return edges


def cantor_dust(level: int = 3, ratio: float = 1 / 3, dim: int = 2) -> np.ndarray:
    """Corners of the planar Cantor dust: 2^(dim * level) points, bi-Lipschitz to an ultrametric."""
    pts = np.zeros((1, dim))
    size = 1.0
    for _ in range(level):
        size_next = size * ratio
        shifts = np.array(np.meshgrid(*[[0.0, size - size_next]] * dim)).reshape(dim, -1).T
        pts = (pts[:, None, :] + shifts[None, :, :]).reshape(-1, dim)
        size = size_next
    return pts + size / 2


__all__ = [
    "cantor_dust",
    "clade_haar_basis",
    "is_ultrametric",
    "log_resistance_ultrametric",
    "reroot",
    "resistance_bound",
    "single_linkage_ultrametric",
    "tree_covariance",
    "tree_diffusion",
]
