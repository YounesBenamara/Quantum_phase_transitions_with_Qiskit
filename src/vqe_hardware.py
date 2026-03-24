"""
VQE on the REAL IBM Marrakesh QPU.

Runs the Transverse-Field Ising Model (TFIM) VQE using the Hardware-
Efficient Ansatz on the ibm_marrakesh quantum processor via
qiskit-ibm-runtime.

Usage
-----
    export IBMQ_TOKEN="your_token_here"   # or use saved credentials
    cd src/
    python vqe_hardware.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from qiskit_ibm_runtime import QiskitRuntimeService, Session, EstimatorV2 as Estimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from exact_solver import get_hamiltonian, compute_lowest_energies
from vqe_solver import build_hea, run_vqe


# ═══════════════════════════════════════════════════════════════════
#                       CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

BACKEND_NAME = "ibm_marrakesh"     # Real QPU target

# --- Physics ---
N        = 4       # Number of spins (qubits)
g        = 0.2     # Transverse field strength (g=1 is critical point)
N_LAYERS = 1        # Ansatz layers — keep low for real hardware

# --- Ansatz — Hardcoded to Real HEA (Ry-only) for shallower circuit ---

# --- Optimizer ---
OPTIMIZER          = "spsa"     # SPSA is noise-resilient (no gradient needed)
SPSA_MAXITER       = 30        # Conservative iteration count for QPU
SPSA_LEARNING_RATE = 0.05
SPSA_PERTURBATION  = 0.1


# --- Shots ---
SHOTS = 4096        # Shots per circuit evaluation

# --- Early stopping ---
CONV_WINDOW = 10    # Plateau detection: window size
CONV_TOL    = 1e-3  # Plateau detection: max std-dev

# --- Output ---
FIG_DIR = "figures/vqe_figs/hardware"


# ═══════════════════════════════════════════════════════════════════

np.random.seed(42)
os.makedirs(FIG_DIR, exist_ok=True)


# ── 1. Connect to IBM Quantum ────────────────────────────────────

token = os.environ.get("IBMQ_TOKEN")

if token:
    service = QiskitRuntimeService(channel="ibm_quantum", token=token)
    print(f"Authenticated via IBMQ_TOKEN environment variable.")
else:
    # Falls back to saved credentials
    # (run QiskitRuntimeService.save_account(...) once beforehand)
    service = QiskitRuntimeService(channel="ibm_quantum")
    print(f"Authenticated via saved credentials.")

backend = service.backend(BACKEND_NAME)
print(f"Backend        : {backend.name}")
print(f"Num qubits     : {backend.num_qubits}")
print(f"Backend status : {'online' if backend.status().operational else 'OFFLINE'}")


# ── 2. Build ansatz & Hamiltonian ─────────────────────────────────

ansatz, params = build_hea(N, N_LAYERS, real=True)
n_params = ansatz.num_parameters
Hamiltonian = get_hamiltonian(N, g)

print(f"\nAnsatz type    : Real HEA")
print(f"Ansatz params  : {n_params}")
print(f"Original depth : {ansatz.depth()}")


# ── 3. Transpile for hardware ─────────────────────────────────────

pm = generate_preset_pass_manager(target=backend.target, optimization_level=3)
ansatz_isa = pm.run(ansatz)

print(f"ISA depth      : {ansatz_isa.depth()}")
print(f"ISA gate count : {dict(ansatz_isa.count_ops())}")

# Remap Hamiltonian to match the transpiled circuit's qubit layout
Hamiltonian_isa = Hamiltonian.apply_layout(layout=ansatz_isa.layout)


# ── 4. Exact reference (classical) ───────────────────────────────

vals_exact, _ = compute_lowest_energies(N, g)
E_exact = vals_exact[0]
print(f"\nExact ground-state energy: {E_exact:.6f}")


# ── 5. Initial parameters ────────────────────────────────────────

PARAMS_FILE = f"{FIG_DIR}/params_N{N}_g{g}_{OPTIMIZER}.npy"

if os.path.exists(PARAMS_FILE):
    initial_params = np.load(PARAMS_FILE)
    print(f"Loaded checkpoint from {PARAMS_FILE}")
else:
    initial_params = np.random.uniform(0, 2 * np.pi, n_params)
    print("Starting from random initial parameters")


# ── 6. Run VQE on real QPU ────────────────────────────────────────

print(f"\n{'═'*50}")
print(f"  Starting VQE on {BACKEND_NAME}")
print(f"  Optimizer: {OPTIMIZER.upper()}, Shots: {SHOTS}")
print(f"  N={N}, g={g}, Layers={N_LAYERS}")
print(f"{'═'*50}\n")

with Session(backend=backend) as session:

    estimator = Estimator(session=session)
    estimator.options.default_shots = SHOTS

    E_vqe, cost_history, converged_early = run_vqe(
        ansatz_isa, Hamiltonian_isa, estimator, initial_params,
        optimizer=OPTIMIZER,
        cobyla_maxiter=COBYLA_MAXITER, cobyla_rhobeg=COBYLA_RHOBEG,
        cobyla_tol=COBYLA_TOL,
        spsa_maxiter=SPSA_MAXITER, spsa_learning_rate=SPSA_LEARNING_RATE,
        spsa_perturbation=SPSA_PERTURBATION,
        conv_window=CONV_WINDOW, conv_tol=CONV_TOL,
    )

print("Session closed.")


# ── 7. Save optimized parameters ─────────────────────────────────

np.save(PARAMS_FILE, cost_history)
print(f"Parameters saved to {PARAMS_FILE}")


# ── 8. Results ────────────────────────────────────────────────────

rel_error = abs(E_vqe - E_exact) / abs(E_exact)

print(f"\n{'─'*50}")
print(f"  Backend       : {BACKEND_NAME} (REAL QPU)")
print(f"  Config        : real HEA, {OPTIMIZER.upper()}, {SHOTS} shots")
print(f"  Exact energy  : {E_exact:.6f}")
if converged_early:
    print(f"  VQE energy    : {E_vqe:.6f}  (plateau mean, early stop)")
else:
    print(f"  VQE energy    : {E_vqe:.6f}  (mean of last {CONV_WINDOW} iters)")
print(f"  Relative error: {rel_error*100:.3f} %")
print(f"  Iterations    : {len(cost_history)}")
print(f"{'─'*50}")


# ── 9. Convergence plot ──────────────────────────────────────────

ansatz_label = "Real HEA"

plt.figure(figsize=(10, 5))
plt.plot(cost_history,
         label=f"E_vqe,  err={rel_error*100:.2f}%", color='#0066CC')

plt.axhline(y=E_exact, color='#FF3300', linestyle='--',
            label=f"Exact: {E_exact:.4f}")

if converged_early:

    plt.axhline(y=E_vqe, color='#00AA44', linestyle=':',
                label=f"Plateau Mean: {E_vqe:.4f}")

    plt.axvline(x=len(cost_history) - 1, color='gray',
                linestyle=':', alpha=0.6, label="Early stop")

plt.xlabel("Iteration")
plt.ylabel(r"Energy $\langle H \rangle$")

plt.title(f"VQE on {BACKEND_NAME} — {ansatz_label}, {OPTIMIZER.upper()}\n"
          f"TFIM  N={N}, g={g}, Layers={N_LAYERS}, Shots={SHOTS}")
plt.legend()
plt.grid(True, alpha=0.3)

fig_path = f"{FIG_DIR}/convergence_N{N}_g{g}_{OPTIMIZER}_{SHOTS}shots.png"
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"\nPlot saved to {fig_path}")
