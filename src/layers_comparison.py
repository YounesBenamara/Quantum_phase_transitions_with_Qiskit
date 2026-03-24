"""
Compare VQE convergence for different numbers of HEA layers.
Produces a single plot with one curve per N_LAYERS value.
"""

from qiskit.primitives import StatevectorEstimator
import numpy as np
import os
import matplotlib.pyplot as plt
from exact_solver import get_hamiltonian, compute_lowest_energies
from vqe_solver import build_hea, run_vqe

if __name__ == "__main__":

    # ── Configuration ──────────────────────────────────────────────
    N = 3
    g = 2
    OPTIMIZER = "cobyla"        # "cobyla" or "spsa"
    REAL_ANSATZ = True

    LAYERS_LIST = [1, 2, 3, 4]  # layers to compare
    
    COBYLA_MAXITER = 120
    COBYLA_RHOBEG  = 0.5
    COBYLA_TOL     = 1e-4

    SPSA_MAXITER       = 100
    SPSA_LEARNING_RATE = 0.05
    SPSA_PERTURBATION  = 0.1

    CONV_WINDOW = 10
    CONV_TOL    = 1e-3

    np.random.seed(42)

    FIG_DIR = "figures/vqe_figs/layers_comparison"
    os.makedirs(FIG_DIR, exist_ok=True)

    # ── Exact reference ────────────────────────────────────────────
    Hamiltonian = get_hamiltonian(N, g)
    vals_exact, _ = compute_lowest_energies(N, g)
    E_exact = vals_exact[0]
    print(f"Exact energy: {E_exact:.6f}")

    # ── Run for each layer count ───────────────────────────────────
    estimator = StatevectorEstimator()

    fig, axes = plt.subplots(len(LAYERS_LIST), 1, figsize=(12, 3 * len(LAYERS_LIST)),
                             sharex=True)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(LAYERS_LIST)))

    for idx, n_layers in enumerate(LAYERS_LIST):

        ansatz, params = build_hea(N, n_layers, real=REAL_ANSATZ)
        n_params = ansatz.num_parameters
        initial_params = np.random.uniform(0, 2 * np.pi, n_params)

        E_vqe, cost_history, converged_early = run_vqe(
            ansatz, Hamiltonian, estimator, initial_params,
            optimizer=OPTIMIZER,
            cobyla_maxiter=COBYLA_MAXITER, cobyla_rhobeg=COBYLA_RHOBEG, cobyla_tol=COBYLA_TOL,
            spsa_maxiter=SPSA_MAXITER, spsa_learning_rate=SPSA_LEARNING_RATE,
            spsa_perturbation=SPSA_PERTURBATION,
            conv_window=CONV_WINDOW, conv_tol=CONV_TOL,
        )

        rel_err = abs(E_vqe - E_exact) / abs(E_exact) * 100
        label = (f"{n_layers} layer{'s' if n_layers > 1 else ''} "
                 f"({n_params} params), E={E_vqe:.4f}, err={rel_err:.2f}%")

        print(f"  Final energy: {E_vqe:.6f}  |  Rel. error: {rel_err:.3f}%  |  "
              f"Iters: {len(cost_history)}")

        ax = axes[idx]
        ax.plot(cost_history, label=label, color=colors[idx], linewidth=1.5)
        ax.axhline(y=E_exact, color='#FF3300', linestyle='--', linewidth=1.2,
                   label=f"Exact: {E_exact:.4f}")
        ax.set_ylabel(r"$\langle H \rangle$")
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)

    # ── Plot ───────────────────────────────────────────────────────
    ansatz_label = "Real HEA" if REAL_ANSATZ else "General HEA"
    axes[-1].set_xlabel("Iteration")
    fig.suptitle(f"Effect of Layers on VQE Convergence ({OPTIMIZER.upper()}, {ansatz_label})\n"
                 f"TFIM  N={N}, g={g}", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/layers_N{N}_g{g}_{OPTIMIZER}.png", dpi=300)
    plt.show()
    print(f"\nPlot saved to {FIG_DIR}/layers_N{N}_g{g}_{OPTIMIZER}.png")

