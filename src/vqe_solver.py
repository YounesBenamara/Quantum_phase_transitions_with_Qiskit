
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
import numpy as np
import os
import matplotlib.pyplot as plt
from exact_solver import get_hamiltonian, compute_lowest_energies




#---CONFIGURATION BLOCK---


OPTIMIZER   = "cobyla"   # "cobyla" or "spsa"
REAL_ANSATZ = True     # True = Ry-only (2N params), False = Rz-Rx-Rz (6N params)
IDEAL       = True  # True = StatevectorEstimator (no noise), False = noisy FakeMarrakesh QPU simulator

N        = 4  # Number of spins
g        = 0.1 # Transverse field force
N_LAYERS = 1    # Number of HEA repetitions — increase for more expressibility

# Optimizer-specific settings
COBYLA_MAXITER = 100
COBYLA_RHOBEG  = 0.5
COBYLA_TOL     = 1e-4

SPSA_MAXITER      = 50 if IDEAL else 25
SPSA_LEARNING_RATE = 0.05
SPSA_PERTURBATION  = 0.1

SHOTS = 10000   # Only used in noisy mode

# Early stopping: stop when the last CONV_WINDOW energies
# fluctuate with std < CONV_TOL (plateau detected)
CONV_WINDOW = 10    # Number of recent iterations to check
CONV_TOL    = 1e-4  # Max std-dev to consider converged

# --- Quick mode (noisy only) ---
PLOT_LAYOUT_ONLY = True   # True = plot qubit layout and exit (skip VQE)

# ══════════════════════════════════════════════════════════════

#Let there be order
np.random.seed(42)


class ConvergedEarly(Exception):
    """Raised when the optimizer has plateaued."""
    def __init__(self, mean_energy, params):
        self.mean_energy = mean_energy
        self.params = params



#---ANSATZ CREATION---


def build_hea(N: int, n_layers: int, real: bool = False) -> tuple[QuantumCircuit, ParameterVector]:
    '''
    Hardware-Efficient Ansatz (HEA) for the TFIM.

    If real=False (general):
        Each layer: Rz-Rx-Rz rotations → CNOT → Rz-Rx-Rz rotations
        → 6N parameters per layer.

    If real=True (real-valued):
        Since the TFIM Hamiltonian is purely real,
        the ground state eigenvector is real-valued.
        Rz gates introduce complex phases that are unnecessary.
        Each layer: Ry rotations → CNOT → Ry rotations
        2N parameters per layer.

    Args:
        N        : number of qubits
        n_layers : number of HEA repetitions
        real     : if True, use Ry-only (real-valued ansatz)

    Returns:
        (circuit, params): parameterized QuantumCircuit and ParameterVector
    '''
    params_per_layer = 2 * N if real else 6 * N
    params = ParameterVector("θ", params_per_layer * n_layers)
    qc = QuantumCircuit(N)

    for layer in range(n_layers):
        offset = params_per_layer * layer

        # Pre-entanglement rotations
        if real:
            for i in range(N):
                qc.ry(params[offset + i], i)
        else:
            for i in range(N):
                qc.rz(params[offset + i], i)
                qc.rx(params[offset + N + i], i)
                qc.rz(params[offset + 2 * N + i], i)

        qc.barrier()

        # Entangling layer (periodic boundary conditions)
        for i in range(N):
            qc.cx(i, (i + 1) % N)

        qc.barrier()

        # Post-entanglement rotations
        if real:
            for i in range(N):
                qc.ry(params[offset + N + i], i)
        else:
            for i in range(N):
                qc.rz(params[offset + 3 * N + i], i)
                qc.rx(params[offset + 4 * N + i], i)
                qc.rz(params[offset + 5 * N + i], i)

        qc.barrier()

    return qc, params


def run_vqe(ansatz, hamiltonian, estimator, initial_params,
            optimizer="cobyla",
            cobyla_maxiter=50, cobyla_rhobeg=0.5, cobyla_tol=1e-4,
            spsa_maxiter=100, spsa_learning_rate=0.05, spsa_perturbation=0.1,
            conv_window=10, conv_tol=1e-4):
    '''
    Run a single VQE optimization.

    Args:
        ansatz:          Parameterized quantum circuit
        hamiltonian:     SparsePauliOp Hamiltonian
        estimator:       Estimator primitive instance
        initial_params:  Initial parameter array
        optimizer:       "cobyla" or "spsa"
        conv_window:     Plateau detection window size
        conv_tol:        Plateau detection std tolerance

    Returns:
        (E_vqe, cost_history, converged_early)
    '''
    cost_history = []

    def cost_func(params):
        pub = (ansatz, [hamiltonian], [params])
        result = estimator.run(pubs=[pub]).result()
        energy = result[0].data.evs[0]
        cost_history.append(energy)
    

        # Plateau-based early stopping
        if len(cost_history) >= conv_window:
            recent = cost_history[-conv_window:]
            if np.std(recent) < conv_tol:
                mean_e = np.mean(recent)
                print(f"\n>>> Converged early: last {conv_window} values have "
                      f"std={np.std(recent):.2e} < tol={conv_tol:.2e}")
                print(f">>> Plateau mean energy: {mean_e:.6f}")
                raise ConvergedEarly(mean_e, params)
        return energy

    converged_early = False
    try:
        if optimizer == "cobyla":
            from scipy.optimize import minimize
            result = minimize(cost_func, initial_params, method="cobyla",
                              options={"maxiter": cobyla_maxiter,
                                       "rhobeg": cobyla_rhobeg,
                                       "tol": cobyla_tol})
        elif optimizer == "spsa":
            from qiskit_algorithms.optimizers import SPSA
            spsa = SPSA(maxiter=spsa_maxiter,
                        learning_rate=spsa_learning_rate,
                        perturbation=spsa_perturbation)
            result = spsa.minimize(fun=cost_func, x0=initial_params)
    except ConvergedEarly as e:
        converged_early = True
        E_vqe = e.mean_energy

    if not converged_early:
        # Use plateau mean instead of result.fun — SPSA's last evaluation
        # can be a perturbation probe, not the actual optimum.
        if len(cost_history) >= conv_window:
            E_vqe = np.mean(cost_history[-conv_window:])
        else:
            E_vqe = np.mean(cost_history)

    return E_vqe, cost_history, converged_early



if __name__ == "__main__":

    # --- Output directories ---
    tag = f"{'real' if REAL_ANSATZ else 'general'}_{'ideal' if IDEAL else 'noisy'}_{OPTIMIZER}"
    FIG_DIR = f"figures/vqe_figs/{tag}"
    os.makedirs(FIG_DIR, exist_ok=True)

    if not IDEAL:
        CHECKPOINT = f"{FIG_DIR}/params_N{N}_g{g}.npy"

    #---ANSATZ & BACKEND SETUP---

    # Build ansatz
    ansatz, params = build_hea(N, N_LAYERS, real=REAL_ANSATZ)
    n_params = ansatz.num_parameters

    # Hamiltonian
    Hamiltonian = get_hamiltonian(N, g)

    if IDEAL:
        # --- Ideal simulation: no noise, no transpilation ---
        from qiskit.primitives import StatevectorEstimator

        print(f"Ansatz depth      : {ansatz.depth()}")
        print(f"Ansatz gate count : {ansatz.count_ops()}")
        print(f"Number of params  : {ansatz.num_parameters}")

        estimator = StatevectorEstimator()
        circuit_to_run = ansatz
        hamiltonian_to_run = Hamiltonian
        shots_label = "ideal"

    else:
        # --- Noisy simulation on FakeMarrakesh ---
        from qiskit_ibm_runtime import EstimatorV2 as Estimator, Batch
        from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        backend = FakeMarrakesh()
        pm = generate_preset_pass_manager(target=backend.target, optimization_level=3)

        ansatz_isa = pm.run(ansatz)

        print(f"Original depth      : {ansatz.depth()}")
        print(f"ISA depth           : {ansatz_isa.depth()}")
        print(f"Original gate count : {ansatz.count_ops()}")
        print(f"ISA gate count      : {ansatz_isa.count_ops()}")

        Hamiltonian_isa = Hamiltonian.apply_layout(layout=ansatz_isa.layout)

        # --- Plot physical qubit layout on QPU ---
        from qiskit.visualization import plot_gate_map

        physical_qubits = ansatz_isa.layout.final_index_layout(filter_ancillas=True)
        print(f"Physical qubits : {physical_qubits}")

        qubit_colors = ['#FF4444' if i in physical_qubits else '#DDDDDD'
                        for i in range(backend.num_qubits)]

        fig_map = plot_gate_map(backend, qubit_color=qubit_colors,
                                qubit_size=28)
        fig_map.suptitle(f"Qubits used on {backend.name}: {physical_qubits}", fontsize=13)
        fig_map.savefig(f"{FIG_DIR}/qubit_layout_N{N}_g{g}.png", dpi=300, bbox_inches='tight')
        print(f"Qubit layout saved to {FIG_DIR}/qubit_layout_N{N}_g{g}.png")

        # --- Plot circuits ---
        fig_circ = ansatz.draw('mpl')
        if fig_circ:
            fig_circ.savefig(f"{FIG_DIR}/circuit_original_N{N}_g{g}.png", dpi=300, bbox_inches='tight')
            print(f"Original circuit saved to {FIG_DIR}/circuit_original_N{N}_g{g}.png")

        fig_isa = ansatz_isa.draw('mpl', idle_wires=False)
        if fig_isa:
            fig_isa.savefig(f"{FIG_DIR}/circuit_isa_N{N}_g{g}.png", dpi=300, bbox_inches='tight')
            print(f"ISA circuit saved to {FIG_DIR}/circuit_isa_N{N}_g{g}.png")

        plt.show()

        if PLOT_LAYOUT_ONLY:
            raise SystemExit(0)

        circuit_to_run = ansatz_isa
        hamiltonian_to_run = Hamiltonian_isa
        shots_label = f"{SHOTS}shots"


    #---EXACT REFERENCE---


    vals_exact, _ = compute_lowest_energies(N, g)
    E_exact = vals_exact[0]


    #---INITIAL PARAMETERS---


    if not IDEAL and os.path.exists(CHECKPOINT):
        initial_params = np.load(CHECKPOINT)
        print(f"Loaded checkpoint params from {CHECKPOINT}")
    else:
        initial_params = np.random.uniform(0, 2 * np.pi, n_params)
        print("Starting from random initial params")


    #---OPTIMIZATION---


    if not IDEAL:
        batch = Batch(backend=backend)
        estimator = Estimator(mode=batch)
        estimator.options.default_shots = SHOTS

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



    #---RESULTS---


    rel_error = abs(E_vqe - E_exact) / abs(E_exact)

    print(f"\n{'─'*40}")
    print(f"  Config        : {tag}")
    print(f"  Exact energy  : {E_exact:.6f}")
    if converged_early:
        print(f"  VQE energy    : {E_vqe:.6f}  (plateau mean over last {CONV_WINDOW} iters)")
    else:
        print(f"  VQE energy    : {E_vqe:.6f}  (mean of last {CONV_WINDOW} iters)")
    print(f"  Relative error: {rel_error*100:.3f} %")
    if converged_early:
        print(f"  Converged     : True (early stop — plateau detected)")
    print(f"  Iterations    : {len(cost_history)}")
    print(f"{'─'*40}")



    #---CONVERGENCE PLOT---

    ansatz_label = "real HEA" if REAL_ANSATZ else "general HEA"
    sim_label = "ideal" if IDEAL else "noisy"
    optim_label = OPTIMIZER.upper()

    plt.figure(figsize=(10, 5))
    plt.plot(cost_history,
             label=f"E_vqe,  err={rel_error*100:.2f}%", color='#0066CC')
    plt.axhline(y=E_exact, color='#FF3300', linestyle='--',
                label=f"Exact Energy: {E_exact:.4f}")
    if converged_early:
        plt.axhline(y=E_vqe, color='#00AA44', linestyle=':',
                    label=f"Plateau Mean: {E_vqe:.4f}")
        plt.axvline(x=len(cost_history) - 1, color='gray',
                    linestyle=':', alpha=0.6, label="Early stop")
    plt.xlabel("Iterations")
    plt.ylabel(r"Energy $\langle H \rangle$")
    plt.title(f"VQE Convergence with {ansatz_label}, {optim_label} algorithm on {sim_label} QPU"
              f" (N={N}, g={g}, Layers={N_LAYERS})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{FIG_DIR}/convergence_N{N}_g{g}_{shots_label}.png", dpi=300)
    plt.show()
