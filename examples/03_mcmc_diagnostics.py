"""
Example 03: MCMC Chain Convergence & Gelman-Rubin Diagnostics
=============================================================
Runs multiple independent MCMC chains starting from distinct initializations 
(Spectral, Agglomerative Ward, and K-Means partitions), computes Gelman-Rubin R-hat 
and Effective Sample Size (ESS), and plots multi-chain convergence traces.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from bgforest.models.bsf import BayesianSpanningForest
from bgforest.datasets.synthetic import make_two_moons
from bgforest.mcmc.diagnostics import compute_gelman_rubin_rhat, compute_effective_sample_size


def main():
    print("==================================================")
    print("   MCMC Chain Convergence & Diagnostic Analysis")
    print("==================================================")

    # 1. Generate Dataset
    X, y_true = make_two_moons(n_samples=180, noise=0.12, random_state=42)

    # 2. Configure 3 Distinct Initializations for Chains
    print("Configuring 3 independent MCMC chains from distinct initializations...")
    
    # Chain 1: Spectral Partition Init
    bsf1 = BayesianSpanningForest(n_clusters=2, n_iter=500, burn_in=100, random_state=42)
    bsf1.fit(X)

    # Chain 2: Agglomerative Ward Partition Init
    ward_init = AgglomerativeClustering(n_clusters=2).fit_predict(X)
    bsf2 = BayesianSpanningForest(n_clusters=2, n_iter=500, burn_in=100, random_state=100)
    bsf2.fit(X, initial_partition=ward_init)

    # Chain 3: K-Means Partition Init
    km_init = KMeans(n_clusters=2, random_state=2026, n_init=10).fit_predict(X)
    bsf3 = BayesianSpanningForest(n_clusters=2, n_iter=500, burn_in=100, random_state=2026)
    bsf3.fit(X, initial_partition=km_init)

    chains_traces = [
        bsf1.log_posterior_trace_,
        bsf2.log_posterior_trace_,
        bsf3.log_posterior_trace_
    ]

    # 3. Compute Gelman-Rubin R-hat and Effective Sample Size (ESS)
    post_burn_in_chains = [trace[100:] for trace in chains_traces]
    rhat = compute_gelman_rubin_rhat(post_burn_in_chains)
    ess_chain1 = compute_effective_sample_size(post_burn_in_chains[0])

    print("\n--- Diagnostic Results ---")
    print(f"Gelman-Rubin R-hat: {rhat:.4f}  (R-hat < 1.05 indicates convergence across chains)")
    print(f"Chain 1 Effective Sample Size (ESS): {ess_chain1:.1f} / {len(post_burn_in_chains[0])} post-burnin samples")
    print(f"Chain 1 Acceptance Rate: {bsf1.acceptance_rate_:.1%}")
    print(f"Chain 2 Acceptance Rate: {bsf2.acceptance_rate_:.1%}")
    print(f"Chain 3 Acceptance Rate: {bsf3.acceptance_rate_:.1%}")

    # 4. Plot Multi-Chain Convergence Traces
    fig, ax = plt.subplots(figsize=(10, 5))
    
    colors = ["#1f77b4", "#ff7f0e", "#d62728"]
    linestyles = ["-", "--", ":"]
    linewidths = [2.2, 1.8, 1.8]
    alphas = [0.85, 0.85, 0.95]
    labels = [
        "Chain 1 (Spectral Init, seed=42)",
        "Chain 2 (Agglomerative Ward Init, seed=100)",
        "Chain 3 (K-Means Init, seed=2026)"
    ]

    for idx, trace in enumerate(chains_traces):
        ax.plot(
            trace,
            color=colors[idx],
            linestyle=linestyles[idx],
            linewidth=linewidths[idx],
            alpha=alphas[idx],
            label=labels[idx]
        )

    ax.axvline(100, color="black", linestyle="-.", linewidth=1.5, label="Burn-in Cutoff (100 iters)")
    ax.set_title(f"Multi-Chain MCMC Posterior Convergence (Gelman-Rubin R-hat = {rhat:.3f})", fontsize=13, fontweight="bold")
    ax.set_xlabel("MCMC Iteration", fontsize=11)
    ax.set_ylabel("Unnormalized Log-Posterior", fontsize=11)
    ax.legend(frameon=True, loc="lower right", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    output_png = "figs/mcmc_multi_chain_diagnostics.png"
    plt.savefig(output_png, dpi=200)
    print(f"\n[Success] Multi-chain convergence figure saved to {output_png}")


if __name__ == "__main__":
    main()
