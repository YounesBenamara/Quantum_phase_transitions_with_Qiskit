
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
import numpy as np
import os
import matplotlib.pyplot as plt
from exact_solver import get_hamiltonian, compute_lowest_energies




#---CONFIGURATION BLOCK---


OPTIMIZER   = "cobyla"   # "cobyla" or "spsa"
REAL_ANSATZ = True     # True = Ry-only (2N params), False = Rz-Rx-Rz (6N params)
IDEAL       = True     # True = StatevectorEstimator (no noise), False = noisy FakeMarrakesh QPU simulator

N        = 3    # Number of spins
g        = 0.1 # Transverse field force
N_LAYERS = 1    # Number of HEA repetitions — increase for more expressibility

# Optimizer-specific settings
COBYLA_MAXITER = 50
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

# ══════════════════════════════════════════════════════════════

#Let there be order
np.random.seed(42)


class ConvergedEarly(Exception):
    """Raised when the optimizer has plateaued."""
    def __init__(self, mean_energy, params):
        self.mean_energy = mean_energy
        self.params = params

# --- Output directories ---
tag = f"{'real' if REAL_ANSATZ else 'general'}_{'ideal' if IDEAL else 'noisy'}_{OPTIMIZER}"
FIG_DIR = f"figures/vqe_figs/{tag}"
os.makedirs(FIG_DIR, exist_ok=True)

#Storing parameters to not start from scratch with the noisy fake backend
if not IDEAL:
    CHECKPOINT = f"{FIG_DIR}/params_N{N}_g{g}.npy"



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



if __name__ == "__main__":

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

        circuit_to_run = ansatz_isa
        hamiltonian_to_run = Hamiltonian_isa
        shots_label = f"{SHOTS}shots"



    #---COST FUNCTION---


    cost_history_dict = {
        "prev_vector": None,
        "iters": 0,
        "cost_history": [],
    }


    def cost_func(params, ansatz, hamiltonian, estimator):
        """Return estimate of energy from estimator

        Parameters:
            params (ndarray): Array of ansatz parameters
            ansatz (QuantumCircuit): Parameterized ansatz circuit
            hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
            estimator: Estimator primitive instance

        Returns:
            float: Energy estimate
        """
        pub = (ansatz, [hamiltonian], [params])
        result = estimator.run(pubs=[pub]).result()
        energy = result[0].data.evs[0]

        cost_history_dict["iters"] += 1
        cost_history_dict["prev_vector"] = params
        cost_history_dict["cost_history"].append(energy)
        print(f"Iters. done: {cost_history_dict['iters']} [Current cost: {energy}]")

        # --- Plateau-based early stopping ---
        # looking at the ten latest values and if their standard deviation is under 1e-4
        history = cost_history_dict["cost_history"]
        if len(history) >= CONV_WINDOW:
            recent = history[-CONV_WINDOW:]
            if np.std(recent) < CONV_TOL:
                mean_e = np.mean(recent)
                print(f"\n>>> Converged early: last {CONV_WINDOW} values have "
                      f"std={np.std(recent):.2e} < tol={CONV_TOL:.2e}")
                print(f">>> Plateau mean energy: {mean_e:.6f}")
                raise ConvergedEarly(mean_e, params)

        return energy


    #---EXACT REFERENCE---


    vals_exact, _ = compute_lowest_energies(N, g)
    E_exact = vals_exact[0]
    print(f"  Exact energy  : {E_exact:.6f}")


    #---INITIAL PARAMETERS---


    if not IDEAL and os.path.exists(CHECKPOINT):
        initial_params = np.load(CHECKPOINT)
        print(f"Loaded checkpoint params from {CHECKPOINT}")
    else:
        initial_params = np.random.uniform(0, 2 * np.pi, n_params)
        print("Starting from random initial params")



    #---OPTIMIZATION---


    converged_early = False

    if IDEAL:
        # --- Ideal: no batch needed ---
        try:
            if OPTIMIZER == "cobyla":
                from scipy.optimize import minimize

                result = minimize(
                    cost_func,
                    initial_params,
                    args=(circuit_to_run, hamiltonian_to_run, estimator),
                    method="cobyla",
                    options={"maxiter": COBYLA_MAXITER, "rhobeg": COBYLA_RHOBEG, "tol": COBYLA_TOL},
                )

            elif OPTIMIZER == "spsa":
                from qiskit_algorithms.optimizers import SPSA

                spsa = SPSA(maxiter=SPSA_MAXITER, learning_rate=SPSA_LEARNING_RATE,
                            perturbation=SPSA_PERTURBATION)

                def objective(params):
                    return cost_func(params, circuit_to_run, hamiltonian_to_run, estimator)

                result = spsa.minimize(fun=objective, x0=initial_params)

        except ConvergedEarly as e:
            converged_early = True
            E_converged = e.mean_energy
            best_params = e.params

    else:
        # --- Noisy: wrap in Batch ---
        batch = Batch(backend=backend)
        estimator = Estimator(mode=batch)
        estimator.options.default_shots = SHOTS

        try:
            if OPTIMIZER == "cobyla":
                from scipy.optimize import minimize

                result = minimize(
                    cost_func,
                    initial_params,
                    args=(circuit_to_run, hamiltonian_to_run, estimator),
                    method="cobyla",
                    options={"maxiter": COBYLA_MAXITER, "rhobeg": COBYLA_RHOBEG, "tol": COBYLA_TOL},
                )

            elif OPTIMIZER == "spsa":
                from qiskit_algorithms.optimizers import SPSA

                spsa = SPSA(maxiter=SPSA_MAXITER, learning_rate=SPSA_LEARNING_RATE,
                            perturbation=SPSA_PERTURBATION)

                def objective(params):
                    return cost_func(params, circuit_to_run, hamiltonian_to_run, estimator)

                result = spsa.minimize(fun=objective, x0=initial_params)

        except ConvergedEarly as e:
            converged_early = True
            E_converged = e.mean_energy
            best_params = e.params

        batch.close()



    if converged_early:
        E_vqe = E_converged
        best_params_final = best_params
    else:
        E_vqe = result.fun
        best_params_final = result.x

    if not IDEAL:
        np.save(CHECKPOINT, best_params_final)
        print(f"Saved optimized params to {CHECKPOINT}")



    #---RESULTS---


    rel_error = abs(E_vqe - E_exact) / abs(E_exact)

    print(f"\n{'─'*40}")
    print(f"  Config        : {tag}")
    print(f"  Exact energy  : {E_exact:.6f}")
    if converged_early:
        print(f"  VQE energy    : {E_vqe:.6f}  (plateau mean over last {CONV_WINDOW} iters)")
    else:
        print(f"  VQE energy    : {E_vqe:.6f}")
    print(f"  Relative error: {rel_error*100:.3f} %")
    if converged_early:
        print(f"  Converged     : True (early stop — plateau detected)")
    elif hasattr(result, 'success'):
        print(f"  Converged     : {result.success}")
    print(f"  Iterations    : {cost_history_dict['iters']}")
    print(f"{'─'*40}")



    #---CONVERGENCE PLOT---

    ansatz_label = "Real HEA" if REAL_ANSATZ else "General HEA"
    sim_label = "Ideal" if IDEAL else "Noisy"
    optim_label = OPTIMIZER.upper()

    plt.figure(figsize=(10, 5))
    plt.plot(cost_history_dict["cost_history"],
             label=f"VQE path ({optim_label}, {sim_label})", color='#0066CC')
    plt.axhline(y=E_exact, color='#FF3300', linestyle='--',
                label=f"Exact Energy: {E_exact:.4f}")
    if converged_early:
        plt.axhline(y=E_vqe, color='#00AA44', linestyle=':',
                    label=f"Plateau Mean: {E_vqe:.4f}")
        plt.axvline(x=cost_history_dict["iters"] - 1, color='gray',
                    linestyle=':', alpha=0.6, label="Early stop")
    plt.xlabel("Iterations")
    plt.ylabel(r"Energy $\langle H \rangle$")
    plt.title(f"VQE Convergence {optim_label} {sim_label} ({ansatz_label}) "
              f"- TFIM (N={N}, g={g}, Layers={N_LAYERS})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{FIG_DIR}/convergence_N{N}_g{g}_{shots_label}.png", dpi=300)
    plt.show()
