"""
Compare VQE convergence for different transverse field strengths g.
Produces a subplot per g value showing the convergence curve.
"""

from qiskit.primitives import StatevectorEstimator
import numpy as np
import os
import matplotlib.pyplot as plt
from exact_solver import get_hamiltonian, compute_lowest_energies
from vqe_solver import build_hea, run_vqe

if __name__ == "__main__":

    # ── Configuration 
    N = 3
    N_LAYERS = 1
    OPTIMIZER = "spsa"        # "cobyla" or "spsa"
    REAL_ANSATZ = True
    IDEAL = True                # True = StatevectorEstimator, False = noisy FakeMarrakesh

    g_LIST = [0, 0.1, 0.5, 1, 1.5, 2]   # transverse field strengths to compare

    COBYLA_MAXITER = 120
    COBYLA_RHOBEG  = 0.5
    COBYLA_TOL     = 1e-4

    SPSA_MAXITER       = 100
    SPSA_LEARNING_RATE = 0.05
    SPSA_PERTURBATION  = 0.1

    CONV_WINDOW = 10
    CONV_TOL    = 1e-3

    np.random.seed(42)

    FIG_DIR = "figures/vqe_figs/g_comparison"
    os.makedirs(FIG_DIR, exist_ok=True)

    # ── Backend setup ──────────────────────────────────────────────
    if IDEAL:
        estimator = StatevectorEstimator()
    else:
        from qiskit_ibm_runtime import EstimatorV2 as Estimator, Batch
        from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        backend = FakeMarrakesh()
        pm = generate_preset_pass_manager(target=backend.target, optimization_level=3)

    # ── Build ansatz (same for all g values) ───────────────────────
    ansatz, params = build_hea(N, N_LAYERS, real=REAL_ANSATZ)
    n_params = ansatz.num_parameters

    # ── Run for each g value ───────────────────────────────────────
    fig, axes = plt.subplots(len(g_LIST), 1, figsize=(12, 3 * len(g_LIST)),
                             sharex=True)
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(g_LIST)))

    for idx, g in enumerate(g_LIST):

        Hamiltonian = get_hamiltonian(N, g)
        vals_exact, _ = compute_lowest_energies(N, g)
        E_exact = vals_exact[0]

        initial_params = np.random.uniform(0, 2 * np.pi, n_params)

        if IDEAL:
            circuit_to_run = ansatz
            hamiltonian_to_run = Hamiltonian
        else:
            ansatz_isa = pm.run(ansatz)
            hamiltonian_to_run = Hamiltonian.apply_layout(layout=ansatz_isa.layout)
            circuit_to_run = ansatz_isa
            batch = Batch(backend=backend)
            estimator = Estimator(mode=batch)
            estimator.options.default_shots = 10000

        E_vqe, cost_history, converged_early = run_vqe(
            circuit_to_run, hamiltonian_to_run, estimator, initial_params,
            optimizer=OPTIMIZER,
            cobyla_maxiter=COBYLA_MAXITER, cobyla_rhobeg=COBYLA_RHOBEG, cobyla_tol=COBYLA_TOL,
            spsa_maxiter=SPSA_MAXITER, spsa_learning_rate=SPSA_LEARNING_RATE,
            spsa_perturbation=SPSA_PERTURBATION,
            conv_window=CONV_WINDOW, conv_tol=CONV_TOL,
        )

        if not IDEAL:
            batch.close()

        rel_err = abs(E_vqe - E_exact) / abs(E_exact) * 100 if E_exact != 0 else float('inf')
        label = f"g={g} — E_vqe={E_vqe:.4f}, E_exact={E_exact:.4f}, err={rel_err:.2f}%"

        print(f"  g={g},    E_vqe={E_vqe:.6f},    E_exact={E_exact:.6f},    "
              f"err={rel_err:.3f}%,    Iters: {len(cost_history)}")

        ax = axes[idx]
        ax.plot(cost_history, label=label, color=colors[idx], linewidth=1.5)
        ax.axhline(y=E_exact, color='#FF3300', linestyle='--', linewidth=1.2,
                   label="E_Exact")
        ax.set_ylabel(r"$\langle H \rangle$")
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)

    # ── Plot ───────────────────────────────────────────────────────
    ansatz_label = "Real HEA" if REAL_ANSATZ else "General HEA"
    sim_label = "ideal QPU" if IDEAL else "noisy QPU"
    axes[-1].set_xlabel("Iteration")
    fig.suptitle(f"Effect of g on VQE Convergence ({OPTIMIZER.upper()}, {ansatz_label}, {sim_label})\n"
                 f"TFIM  N={N}, Layers={N_LAYERS}", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/g_comparison_N{N}_L{N_LAYERS}_{OPTIMIZER}.png", dpi=300)
    plt.show()
    print(f"\nPlot saved to {FIG_DIR}/g_comparison_N{N}_L{N_LAYERS}_{OPTIMIZER}.png")
