from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

from exact_solver import get_hamiltonian, compute_lowest_energies


# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

REAL_ANSATZ = True
IDEAL = True   # True = noiseless StatevectorEstimator, False = FakeMarrakesh

N = 4
g = 0.1
N_LAYERS = 1

# SPSA settings
SPSA_MAXITER = 100

# SPSA gain sequences:
# ak = A_LR / (k + 1 + A_SHIFT)^ALPHA
# ck = C_PERT / (k + 1)^GAMMA
A_LR = 0.08
A_SHIFT = 10.0
ALPHA = 0.602

C_PERT = 0.10
GAMMA = 0.101

# Only used in fake-backend mode
SHOTS = 4096

# Noise-aware early stopping
PLATEAU_WINDOW = 10
PLATEAU_Z = 1.5
PLATEAU_PATIENCE = 3

SHOW_LAYOUT = True
SHOW_CIRCUITS = True

SEED = 42
np.random.seed(SEED)


# ══════════════════════════════════════════════════════════════
# ANSATZ
# ══════════════════════════════════════════════════════════════

def build_hea(N: int, n_layers: int, real: bool = False):
    params_per_layer = 2 * N if real else 6 * N
    params = ParameterVector("θ", params_per_layer * n_layers)
    qc = QuantumCircuit(N)

    for layer in range(n_layers):
        offset = params_per_layer * layer

        if real:
            for i in range(N):
                qc.ry(params[offset + i], i)
        else:
            for i in range(N):
                qc.rz(params[offset + i], i)
                qc.rx(params[offset + N + i], i)
                qc.rz(params[offset + 2 * N + i], i)

        qc.barrier()

        for i in range(N):
            qc.cx(i, (i + 1) % N)

        qc.barrier()

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


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def scalar(x):
    return float(np.asarray(x).reshape(-1)[0])


def wrap_angles(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


def window_mean_and_se(values, stds):
    values = np.asarray(values, dtype=float)
    stds = np.asarray(stds, dtype=float)
    mu = float(np.mean(values))
    se = float(np.sqrt(np.sum(stds**2)) / len(stds))
    return mu, se


def plateau_test(energies, stds, window, z):
    if len(energies) < 2 * window:
        return False, {}

    prev_vals = np.asarray(energies[-2 * window:-window], dtype=float)
    curr_vals = np.asarray(energies[-window:], dtype=float)
    prev_stds = np.asarray(stds[-2 * window:-window], dtype=float)
    curr_stds = np.asarray(stds[-window:], dtype=float)

    mu_prev, se_prev = window_mean_and_se(prev_vals, prev_stds)
    mu_curr, se_curr = window_mean_and_se(curr_vals, curr_stds)

    improvement = mu_prev - mu_curr
    threshold = z * np.sqrt(se_prev**2 + se_curr**2)

    stop = improvement < threshold
    info = {
        "mu_prev": mu_prev,
        "mu_curr": mu_curr,
        "se_prev": se_prev,
        "se_curr": se_curr,
        "improvement": improvement,
        "threshold": threshold,
    }
    return stop, info


def estimate_triplet(estimator, circuit, observable, theta_plus, theta_minus, theta_center, ideal=False):
    pubs = [
        (circuit, observable, theta_plus),
        (circuit, observable, theta_minus),
        (circuit, observable, theta_center),
    ]
    job = estimator.run(pubs)
    result = job.result()

    e_plus = scalar(result[0].data.evs)
    e_minus = scalar(result[1].data.evs)
    e_center = scalar(result[2].data.evs)

    if ideal:
        s_plus = 0.0
        s_minus = 0.0
        s_center = 0.0
    else:
        s_plus = scalar(result[0].data.stds)
        s_minus = scalar(result[1].data.stds)
        s_center = scalar(result[2].data.stds)

    return (e_plus, s_plus), (e_minus, s_minus), (e_center, s_center)


# ══════════════════════════════════════════════════════════════
# VQE DRIVER
# ══════════════════════════════════════════════════════════════

def run_vqe_spsa_decay(
    ansatz,
    hamiltonian,
    estimator,
    initial_params,
    maxiter=100,
    a_lr=0.08,
    a_shift=10.0,
    alpha=0.602,
    c_pert=0.10,
    gamma=0.101,
    plateau_window=10,
    plateau_z=1.5,
    plateau_patience=3,
    seed=42,
    ideal=False,
):
    rng = np.random.default_rng(seed)

    theta = np.asarray(initial_params, dtype=float).copy()
    theta = wrap_angles(theta)

    n_params = theta.size

    cost_history = []
    std_history = []
    grad_norm_history = []
    ak_history = []
    ck_history = []

    plateau_hits = 0
    converged_early = False

    best_window_mean = np.inf
    best_window_se = np.inf
    best_theta = theta.copy()

    for k in range(maxiter):
        delta = rng.choice([-1.0, 1.0], size=n_params)

        ak = a_lr / ((k + 1 + a_shift) ** alpha)
        ck = c_pert / ((k + 1) ** gamma)

        theta_center = theta.copy()
        theta_plus = wrap_angles(theta_center + ck * delta)
        theta_minus = wrap_angles(theta_center - ck * delta)

        (e_plus, _), (e_minus, _), (e_center, s_center) = estimate_triplet(
            estimator,
            ansatz,
            hamiltonian,
            theta_plus,
            theta_minus,
            theta_center,
            ideal=ideal,
        )

        grad = ((e_plus - e_minus) / (2.0 * ck)) * delta
        grad_norm = float(np.linalg.norm(grad))

        cost_history.append(e_center)
        std_history.append(s_center)
        grad_norm_history.append(grad_norm)
        ak_history.append(ak)
        ck_history.append(ck)

        if len(cost_history) >= plateau_window:
            mu_curr, se_curr = window_mean_and_se(
                cost_history[-plateau_window:],
                std_history[-plateau_window:],
            )
            if mu_curr < best_window_mean:
                best_window_mean = mu_curr
                best_window_se = se_curr
                best_theta = theta_center.copy()

        stop_now, info = plateau_test(
            cost_history,
            std_history,
            window=plateau_window,
            z=plateau_z,
        )

        if stop_now:
            plateau_hits += 1
        else:
            plateau_hits = 0

        msg = (
            f"Iter {k+1:03d}/{maxiter} | "
            f"E={e_center:+.6f}"
        )
        if not ideal:
            msg += f" ± {s_center:.6f}"
        msg += (
            f" | |grad|={grad_norm:.4f} | "
            f"ak={ak:.4e} | ck={ck:.4e}"
        )

        if info:
            msg += (
                f" | Δwin={info['improvement']:+.6f} "
                f"vs thr={info['threshold']:.6f} "
                f"| hits={plateau_hits}/{plateau_patience}"
            )
        print(msg)

        if plateau_hits >= plateau_patience:
            converged_early = True
            break

        theta = wrap_angles(theta_center - ak * grad)

    if np.isfinite(best_window_mean):
        final_energy = best_window_mean
        final_energy_se = best_window_se
    else:
        final_energy = cost_history[-1]
        final_energy_se = std_history[-1]
        best_theta = theta_center.copy()

    return {
        "theta_opt": best_theta,
        "energy_opt": final_energy,
        "energy_opt_se": final_energy_se,
        "cost_history": np.asarray(cost_history, dtype=float),
        "std_history": np.asarray(std_history, dtype=float),
        "grad_norm_history": np.asarray(grad_norm_history, dtype=float),
        "ak_history": np.asarray(ak_history, dtype=float),
        "ck_history": np.asarray(ck_history, dtype=float),
        "converged_early": converged_early,
    }


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ansatz, params = build_hea(N, N_LAYERS, real=REAL_ANSATZ)
    n_params = ansatz.num_parameters
    Hamiltonian = get_hamiltonian(N, g)

    if IDEAL:
        from qiskit.primitives import StatevectorEstimator

        estimator = StatevectorEstimator()
        circuit_to_run = ansatz
        hamiltonian_to_run = Hamiltonian

        print("MODE: IDEAL")
        print(f"Ansatz depth      : {ansatz.depth()}")
        print(f"Ansatz gate count : {dict(ansatz.count_ops())}")
        print(f"Number of params  : {n_params}")

        if SHOW_CIRCUITS:
            print("\nOriginal circuit:")
            display(ansatz.draw("mpl"))

    else:
        from qiskit_ibm_runtime import EstimatorV2 as Estimator
        from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit.visualization import plot_gate_map

        backend = FakeMarrakesh()
        pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
        ansatz_isa = pm.run(ansatz)
        Hamiltonian_isa = Hamiltonian.apply_layout(layout=ansatz_isa.layout)

        print("MODE: FAKE_MARRAKESH")
        print(f"Backend             : {backend.name}")
        print(f"Original depth      : {ansatz.depth()}")
        print(f"ISA depth           : {ansatz_isa.depth()}")
        print(f"Original gate count : {dict(ansatz.count_ops())}")
        print(f"ISA gate count      : {dict(ansatz_isa.count_ops())}")
        print(f"Number of params    : {n_params}")

        if SHOW_LAYOUT:
            physical_qubits = ansatz_isa.layout.final_index_layout(filter_ancillas=True)
            print(f"Physical qubits used: {physical_qubits}")

            qubit_colors = [
                "#FF4444" if i in physical_qubits else "#DDDDDD"
                for i in range(backend.num_qubits)
            ]
            fig_map = plot_gate_map(backend, qubit_color=qubit_colors, qubit_size=28)
            fig_map.suptitle(f"Qubits used on {backend.name}: {physical_qubits}", fontsize=13)
            plt.show()

        if SHOW_CIRCUITS:
            print("\nOriginal circuit:")
            display(ansatz.draw("mpl"))
            print("\nISA circuit:")
            display(ansatz_isa.draw("mpl", idle_wires=False))

        circuit_to_run = ansatz_isa
        hamiltonian_to_run = Hamiltonian_isa

        estimator = Estimator(mode=backend)
        estimator.options.default_shots = SHOTS

    vals_exact, _ = compute_lowest_energies(N, g)
    E_exact = float(vals_exact[0])

    initial_params = 0.05 * np.random.randn(n_params)
    print("Starting from small near-zero initial parameters")

    result = run_vqe_spsa_decay(
        ansatz=circuit_to_run,
        hamiltonian=hamiltonian_to_run,
        estimator=estimator,
        initial_params=initial_params,
        maxiter=SPSA_MAXITER,
        a_lr=A_LR,
        a_shift=A_SHIFT,
        alpha=ALPHA,
        c_pert=C_PERT,
        gamma=GAMMA,
        plateau_window=PLATEAU_WINDOW,
        plateau_z=PLATEAU_Z,
        plateau_patience=PLATEAU_PATIENCE,
        seed=SEED,
        ideal=IDEAL,
    )

    theta_opt = result["theta_opt"]
    E_vqe = result["energy_opt"]
    E_vqe_se = result["energy_opt_se"]
    cost_history = result["cost_history"]
    std_history = result["std_history"]
    converged_early = result["converged_early"]

    rel_error = abs(E_vqe - E_exact) / abs(E_exact)

    print(f"\n{'─'*56}")
    mode_name = "real_ideal_spsa_decay" if IDEAL else "real_fake_marrakesh_spsa_decay"
    print(f"  Config         : {mode_name}")
    print(f"  Exact energy   : {E_exact:.6f}")
    if IDEAL:
        print(f"  VQE energy     : {E_vqe:.6f}")
    else:
        print(f"  VQE energy     : {E_vqe:.6f} ± {E_vqe_se:.6f}")
    print(f"  Relative error : {rel_error*100:.3f} %")
    print(f"  Early stop     : {converged_early}")
    print(f"  Iterations     : {len(cost_history)}")
    print(f"  Max std        : {np.max(std_history):.6f}")
    print(f"{'─'*56}")

    plt.figure(figsize=(10, 5))
    x = np.arange(1, len(cost_history) + 1)

    plt.plot(x, cost_history, marker="o", linewidth=1.5,
             label=f"E_vqe, err={rel_error*100:.2f}%")

    if not IDEAL:
        plt.fill_between(
            x,
            cost_history - std_history,
            cost_history + std_history,
            alpha=0.2,
            label=r"Estimator $\pm 1\sigma$",
        )

    plt.axhline(y=E_exact, linestyle="--", label=f"Exact energy: {E_exact:.6f}")
    plt.axhline(y=E_vqe, linestyle=":", label=f"Reported VQE: {E_vqe:.6f}")

    if converged_early:
        plt.axvline(x=len(cost_history), linestyle=":", alpha=0.6, label="Early stop")

    mode_label = "IDEAL" if IDEAL else "FakeMarrakesh"
    plt.xlabel("Iteration")
    plt.ylabel(r"Energy $\langle H \rangle$")
    plt.title(
        f"VQE on {mode_label} — real HEA, SPSA with decay\n"
        f"(N={N}, g={g}, Layers={N_LAYERS})"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    print("\nOptimized parameters:")
    print(theta_opt)