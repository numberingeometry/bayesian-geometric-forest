"""
Example 03: MCMC Chain Convergence & Gelman-Rubin Diagnostics
=============================================================
Runs multiple independent MCMC chains for Bayesian Spanning Forest sampling,
computes Gelman-Rubin R-hat and Effective Sample Size (ESS), and plots
log-posterior convergence traces.
"""

import numpy as np
import matplotlib.pyplot as plt
from bgforest.models.bsf import BayesianSpanningForest
from bgforest.datasets.synthetic import make_anisotropic_blobs
from bgforest.mcmc.diagnostics import compute_gelman_rubin_rhat, compute_effective_sample_size
from bgforest.viz.posterior_viz import plot_mcmc_trace


def main():
    print("==================================================")
    print("   MCMC Chain Convergence & Diagnostic Analysis")
    print("==================================================")

    # 1. Generate Dataset
    X, y_true = make_anisotropic_blobs(n_samples=180, random_state=42)

    # 2. Run 3 Independent MCMC Chains
    n_chains = 3
    chains_traces = []
    estimators = []

    print(f"Running {n_chains} independent MCMC chains (500 iterations each)...")
    for seed in [42, 100, 2026]:
        bsf = BayesianSpanningForest(
            n_clusters=3, n_iter=500, burn_in=100, random_state=seed
        )
        bsf.fit(X)
        chains_traces.append(bsf.log_posterior_trace_)
        estimators.append(bsf)

    # 3. Compute Gelman-Rubin R-hat and Effective Sample Size (ESS)
    post_burn_in_chains = [trace[100:] for trace in chains_traces]
    rhat = compute_gelman_rubin_rhat(post_burn_in_chains)
    ess_chain1 = compute_effective_sample_size(post_burn_in_chains[0])

    print("\n--- Diagnostic Results ---")
    print(f"Gelman-Rubin R-hat: {rhat:.4f}  (R-hat < 1.05 indicates convergence)")
    print(f"Chain 1 Effective Sample Size (ESS): {ess_chain1:.1f} / {len(post_burn_in_chains[0])} post-burnin samples")

    # 4. Plot Multi-Chain Convergence Traces
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for idx, trace in enumerate(chains_traces):
        ax.plot(trace, color=colors[idx], alpha=0.7, linewidth=1.2, label=f"Chain {idx+1} (seed={42 if idx==0 else (100 if idx==1 else 2026)})")

    ax.axvline(100, color="black", linestyle="--", linewidth=1.5, label="Burn-in (100 iters)")
    ax.set_title(f"Multi-Chain MCMC Convergence (Gelman-Rubin R-hat = {rhat:.3f})", fontsize=12, fontweight="bold")
    ax.set_xlabel("MCMC Iteration", fontsize=11)
    ax.set_ylabel("Unnormalized Log-Posterior", fontsize=11)
    ax.legend(frameon=True, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    output_png = "figs/mcmc_multi_chain_diagnostics.png"
    plt.savefig(output_png, dpi=200)
    print(f"\n[Success] Convergence diagnostic figure saved to {output_png}")


if __name__ == "__main__":
    main()
