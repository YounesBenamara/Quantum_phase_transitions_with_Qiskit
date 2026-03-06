"""
Compare VQE convergence for different numbers of HEA layers.
Produces a single plot with one curve per N_LAYERS value.
"""

from qiskit.primitives import StatevectorEstimator
import numpy as np
import os
import matplotlib.pyplot as plt
from exact_solver import get_hamiltonian, compute_lowest_energies
from vqe_solver import build_hea, ConvergedEarly

# ── Configuration ──────────────────────────────────────────────
N = 3
g = 0
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

plt.figure(figsize=(12, 6))
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(LAYERS_LIST)))

for idx, n_layers in enumerate(LAYERS_LIST):
    print(f"\n{'═'*50}")
    print(f"  Running N_LAYERS = {n_layers}")
    print(f"{'═'*50}")

    ansatz, params = build_hea(N, n_layers, real=REAL_ANSATZ)
    n_params = ansatz.num_parameters
    initial_params = np.random.uniform(0, 2 * np.pi, n_params)

    cost_history = []

    def cost_func(par):
        pub = (ansatz, [Hamiltonian], [par])
        result = estimator.run(pubs=[pub]).result()
        energy = result[0].data.evs[0]
        cost_history.append(energy)

        # Early stopping
        if len(cost_history) >= CONV_WINDOW:
            recent = cost_history[-CONV_WINDOW:]
            if np.std(recent) < CONV_TOL:
                mean_e = np.mean(recent)
                print(f"  → Converged at iter {len(cost_history)}, "
                      f"plateau mean: {mean_e:.6f}")
                raise ConvergedEarly(mean_e, par)
        return energy

    converged_early = False

    try:
        if OPTIMIZER == "cobyla":
            from scipy.optimize import minimize
            res = minimize(cost_func, initial_params, method="cobyla",
                           options={"maxiter": COBYLA_MAXITER,
                                    "rhobeg": COBYLA_RHOBEG,
                                    "tol": COBYLA_TOL})
        elif OPTIMIZER == "spsa":
            from qiskit_algorithms.optimizers import SPSA
            spsa = SPSA(maxiter=SPSA_MAXITER,
                        learning_rate=SPSA_LEARNING_RATE,
                        perturbation=SPSA_PERTURBATION)
            res = spsa.minimize(fun=cost_func, x0=initial_params)

    except ConvergedEarly as e:
        converged_early = True
        E_vqe = e.mean_energy

    

    rel_err = abs(E_vqe - E_exact) / abs(E_exact) * 100
    label = (f"{n_layers} layer{'s' if n_layers > 1 else ''} "
             f"({n_params} params) — E={E_vqe:.4f}, err={rel_err:.2f}%")
   
    print(f"  Final energy: {E_vqe:.6f}  |  Rel. error: {rel_err:.3f}%  |  "
          f"Iters: {len(cost_history)}")

    plt.plot(cost_history, label=label, color=colors[idx], linewidth=1.5)

# ── Plot ───────────────────────────────────────────────────────
plt.axhline(y=E_exact, color='#FF3300', linestyle='--', linewidth=1.5,
            label=f"Exact: {E_exact:.4f}")

ansatz_label = "Real HEA" if REAL_ANSATZ else "General HEA"
plt.xlabel("Iteration")
plt.ylabel(r"Energy $\langle H \rangle$")
plt.title(f"Effect of Layers on VQE Convergence ({OPTIMIZER.upper()}, {ansatz_label})\n"
          f"TFIM  N={N}, g={g}")
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/layers_N{N}_g{g}_{OPTIMIZER}.png", dpi=300)
plt.show()
print(f"\nPlot saved to {FIG_DIR}/layers_N{N}_g{g}_{OPTIMIZER}.png")
