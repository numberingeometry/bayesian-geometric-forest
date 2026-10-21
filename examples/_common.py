"""Shared style and datasets for the example scripts (all figures are SVG)."""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.datasets import load_iris, make_blobs, make_circles, make_moons  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "figs"
FIGS.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "svg.fonttype": "none",
        "svg.hashsalt": "bgforest",
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.edgecolor": "#d1d5db",
        "axes.labelcolor": "#374151",
        "xtick.color": "#6b7280",
        "ytick.color": "#6b7280",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def save(fig, name: str) -> Path:
    path = FIGS / name
    fig.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None})
    preview = os.environ.get("BGF_PREVIEW_DIR")  # optional raster preview for local review only
    if preview:
        fig.savefig(Path(preview) / (Path(name).stem + ".png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1024:.0f} KB)")
    return path


def showcase(n: int = 360, seed: int = 7):
    """Two interleaved moons, a compact blob, a thin bridge to it, and a few outliers."""
    rng = np.random.default_rng(seed)
    Xm, ym = make_moons(int(n * 0.74), noise=0.055, random_state=seed)
    Xb = rng.normal([2.75, 1.25], 0.13, (int(n * 0.21), 2))
    t = np.linspace(0.18, 0.82, 7)[:, None]
    Xr = (1 - t) * np.array([2.05, 0.55]) + t * np.array([2.7, 1.2]) + rng.normal(0, 0.03, (7, 2))
    n_out = n - len(Xm) - len(Xb) - len(Xr)
    Xo = rng.uniform([-1.3, -1.0], [3.3, 1.8], (n_out, 2))
    X = np.vstack([Xm, Xb, Xr, Xo])
    y = np.concatenate([ym, np.full(len(Xb), 2), np.full(len(Xr), -2), np.full(n_out, -1)])
    return X, y


def spirals(n: int = 300, noise: float = 0.22, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.sqrt(rng.uniform(0.08, 1.0, n // 2)) * 2.6 * np.pi
    arms = []
    for k in range(2):
        a = t + k * np.pi
        arms.append(np.c_[t * np.cos(a), t * np.sin(a)] + rng.normal(0, noise, (n // 2, 2)))
    return np.vstack(arms), np.repeat([0, 1], n // 2)


def benchmark_datasets(seed: int):
    X_moons, y_moons = make_moons(300, noise=0.08, random_state=seed)
    X_circ, y_circ = make_circles(300, noise=0.05, factor=0.5, random_state=seed)
    X_blob, y_blob = make_blobs(300, centers=[[0, 0], [3, 0], [1.5, 2.6]], cluster_std=[0.5, 0.9, 0.6], random_state=seed)
    X_anis = X_blob @ np.array([[0.6, -0.64], [-0.41, 0.85]])
    X_sp, y_sp = spirals(300, seed=seed)
    rng = np.random.default_rng(100 + seed)
    X_mo, y_mo = make_moons(285, noise=0.08, random_state=seed)
    X_out = rng.uniform([-1.5, -1.0], [2.5, 1.5], (15, 2))
    iris = load_iris()
    return {
        "Two moons": (X_moons, y_moons),
        "Moons + 5% outliers": (np.vstack([X_mo, X_out]), np.concatenate([y_mo, np.full(15, -1)])),
        "Circles": (X_circ, y_circ),
        "Stretched blobs": (X_anis, y_blob),
        "Spirals": (X_sp, y_sp),
        "Iris (real)": (iris.data, iris.target),
    }
