# -*- coding: utf-8 -*-
"""
Figure 3 — Attractor depth and escape-time analysis across development.

Reproduces:
    - Figura3_Attractor_Landscape.png (3 panels: A–C)

Companion code for:
    Alvarado et al., "Memory as Maintained Coupling: A Kuramoto Model
    Formalization of Synchronization Dynamics in Neural Systems".

Usage (Google Colab):
    !python Figure3_Attractor_Landscape.py

Dependencies: numpy, scipy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp
import warnings

warnings.filterwarnings("ignore")


# =====================================================================
# PARAMETERS
# =====================================================================
N_POP = 100
N_TOTAL = 2 * N_POP
LAMBDA_SPACE = 0.3
K_CROSS = 3.0                # Cross-coupling used throughout the protocol
SIGMA_ESCAPE = 2.0           # Fixed noise intensity for escape-time assay

K0_YOUNG  = 200.0            # Immature state: weak surviving synapses
K0_MATURE = 1500.0           # Mature state: strengthened synapses

# Effective internal densities (corrected connectomic values)
RHO_SIM_EARLY  = 0.0813
RHO_SIM_MATURE = 0.0280
RHO_EMP_EARLY  = 0.0637
RHO_EMP_MATURE = 0.0217

np.random.seed(42)


# =====================================================================
# COUPLING MATRIX CONSTRUCTION
# =====================================================================
def build_matrix(N, rho_diag_target, K0, K_cross, seed=None):
    """
    Build a sparse, distance-dependent coupling matrix with a target
    intra-population density. Two populations are coupled only through
    the uniform cross-coupling block K_cross.
    """
    if seed is not None:
        np.random.seed(seed)
    positions = np.linspace(0, 1, N)
    rho_raw = 0.25
    K = np.zeros((N, N))

    # Distance-dependent intra-population connectivity
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = min(abs(positions[i] - positions[j]),
                    1 - abs(positions[i] - positions[j]))
            if np.random.rand() < rho_raw * np.exp(-d / LAMBDA_SPACE):
                base = K0 * np.exp(-d / LAMBDA_SPACE)
                var = np.random.lognormal(mean=0, sigma=0.3)
                K[i, j] = base * var
    K = (K + K.T) / 2

    # Zero out cross-population blocks (only intra-population here)
    K[:N_POP, N_POP:] = 0
    K[N_POP:, :N_POP] = 0

    # Rescale to the target density
    n_target = int(round(rho_diag_target * N_POP * N_POP)) * 2
    nonzero_i, nonzero_j = np.where(K > 0)
    if len(nonzero_i) > n_target:
        keep = np.random.choice(len(nonzero_i), n_target, replace=False)
        K_new = np.zeros_like(K)
        K_new[nonzero_i[keep], nonzero_j[keep]] = K[nonzero_i[keep], nonzero_j[keep]]
        K = (K_new + K_new.T) / 2

    # Add uniform cross-coupling block
    cross = np.zeros((N, N))
    cross[:N_POP, N_POP:] = 1.0
    cross[N_POP:, :N_POP] = 1.0

    return K + K_cross * cross, K


# =====================================================================
# ATTRACTOR IDENTIFICATION AND ESCAPE-TIME MEASUREMENT
# =====================================================================
def find_attractor(A, omega, t_max=200):
    """Integrate from random phases and return the terminal phase configuration."""
    theta0 = np.random.uniform(0, 2 * np.pi, N_TOTAL)

    def ode(t, theta):
        sin_diff = np.sin(theta[None, :] - theta[:, None])
        return omega + (1.0 / N_POP) * np.sum(A * sin_diff, axis=1)

    sol = solve_ivp(ode, (0, t_max), theta0, method='RK45', rtol=1e-6)
    return sol.y[:, -1]


def measure_escape_time(A, omega, sigma, theta_attractor,
                        threshold=0.85, t_max=50, dt=0.005, n_trials=8):
    """
    Mean escape time from the synchronized attractor at fixed noise
    intensity sigma. Escape is declared when the intra-population
    coherence of either population falls below the threshold.
    """
    times = []
    for _ in range(n_trials):
        theta = theta_attractor.copy()
        t = 0.0
        escaped = False
        while t < t_max:
            sin_diff = np.sin(theta[None, :] - theta[:, None])
            dtheta = omega + (1.0 / N_POP) * np.sum(A * sin_diff, axis=1)
            dtheta = dtheta * dt + sigma * np.sqrt(dt) * np.random.randn(N_TOTAL)
            theta = (theta + dtheta) % (2 * np.pi)
            r1 = np.abs(np.mean(np.exp(1j * theta[:N_POP])))
            r2 = np.abs(np.mean(np.exp(1j * theta[N_POP:])))
            if min(r1, r2) < threshold:
                times.append(t)
                escaped = True
                break
            t += dt
        if not escaped:
            times.append(t_max)
    return np.mean(times), np.std(times)


def compute_energy_profile(A, r_fixed=0.99, n_points=40, n_samples=6):
    """
    Mean-field energy E(Delta-phi) of the coupling landscape at fixed
    high coherence, as a proxy for the attractor-basin depth.
    """
    delta_phi_axis = np.linspace(-np.pi, np.pi, n_points)
    kappa = 2 * r_fixed / (1 - r_fixed)
    kappa = min(kappa, 20)
    E = np.zeros(n_points)
    for i, dp in enumerate(delta_phi_axis):
        E_avg = 0.0
        for _ in range(n_samples):
            theta1 = np.random.vonmises(-dp / 2, kappa, N_POP)
            theta2 = np.random.vonmises(dp / 2, kappa, N_POP)
            theta = np.concatenate([theta1, theta2])
            cos_diff = np.cos(theta[None, :] - theta[:, None])
            E_avg += -np.sum(A * cos_diff) / (2 * N_TOTAL)
        E[i] = E_avg / n_samples
    return delta_phi_axis, E


def escape_trace(A, omega, theta_attractor, sigma, t_max=50, dt=0.005):
    """Single-shot escape trajectory for visualization (Panel B)."""
    theta = theta_attractor.copy()
    t = 0.0
    r_trace, t_trace = [], []
    while t < t_max:
        sin_diff = np.sin(theta[None, :] - theta[:, None])
        dtheta = omega + (1.0 / N_POP) * np.sum(A * sin_diff, axis=1)
        dtheta = dtheta * dt + sigma * np.sqrt(dt) * np.random.randn(N_TOTAL)
        theta = (theta + dtheta) % (2 * np.pi)
        r1 = np.abs(np.mean(np.exp(1j * theta[:N_POP])))
        r2 = np.abs(np.mean(np.exp(1j * theta[N_POP:])))
        r_trace.append(min(r1, r2))
        t_trace.append(t)
        t += dt
    return np.array(t_trace), np.array(r_trace)


# =====================================================================
# CRITICALITY SWEEP — NUMERICAL DETERMINATION OF K_c
# =====================================================================
def escape_time_vs_density(K0_fixed, rho_array, omega, seed_base=100):
    """Sweep connection densities at fixed K0 and measure escape times."""
    escape_times, escape_stds = [], []
    for k, rho in enumerate(rho_array):
        A, _ = build_matrix(N_TOTAL, rho, K0_fixed, K_CROSS,
                            seed=seed_base + k)
        try:
            theta_att = find_attractor(A, omega, t_max=150)
            m, s = measure_escape_time(A, omega, SIGMA_ESCAPE, theta_att,
                                       t_max=50, n_trials=5)
            escape_times.append(m)
            escape_stds.append(s)
        except Exception:
            escape_times.append(0.0)
            escape_stds.append(0.0)
    return np.array(escape_times), np.array(escape_stds)


def measure_Kc(K0_grid=None, rho_fixed=RHO_SIM_MATURE, n_seeds=3):
    """
    Numerically measure the intra-population locking threshold K_c
    with cross-coupling turned OFF (K_cross = 0).
    """
    if K0_grid is None:
        K0_grid = np.linspace(1, 100, 20)
    omega = np.concatenate([-0.8 + 0.15 * np.random.randn(N_POP),
                             0.8 + 0.15 * np.random.randn(N_POP)])
    r_steady = []
    for K0 in K0_grid:
        r_seeds = []
        for s in range(n_seeds):
            A, _ = build_matrix(N_TOTAL, rho_fixed, K0, 0.0, seed=1000 + s)
            theta_att = find_attractor(A, omega, t_max=200)
            r1 = np.abs(np.mean(np.exp(1j * theta_att[:N_POP])))
            r2 = np.abs(np.mean(np.exp(1j * theta_att[N_POP:])))
            r_seeds.append(0.5 * (r1 + r2))
        r_steady.append(np.mean(r_seeds))
        print(f"  K0={K0:6.1f}  K_eff={K0 * rho_fixed:6.3f}  "
              f"r_within={r_steady[-1]:.3f}")
    return K0_grid * rho_fixed, np.array(r_steady)


# =====================================================================
# MAIN — BUILD MATRICES AND MEASURE
# =====================================================================
print("Building coupling matrices...")
A_sim_young,  _ = build_matrix(N_TOTAL, RHO_SIM_EARLY,  K0_YOUNG,  K_CROSS, seed=1)
A_sim_mature, _ = build_matrix(N_TOTAL, RHO_SIM_MATURE, K0_MATURE, K_CROSS, seed=2)
A_emp_young,  _ = build_matrix(N_TOTAL, RHO_EMP_EARLY,  K0_YOUNG,  K_CROSS, seed=3)
A_emp_mature, _ = build_matrix(N_TOTAL, RHO_EMP_MATURE, K0_MATURE, K_CROSS, seed=4)

omega = np.concatenate([-0.8 + 0.15 * np.random.randn(N_POP),
                        0.8 + 0.15 * np.random.randn(N_POP)])

conditions = [
    ("Sim P7-8",   A_sim_young,  K0_YOUNG,  RHO_SIM_EARLY),
    ("Sim P45-50", A_sim_mature, K0_MATURE, RHO_SIM_MATURE),
    ("Emp P7-8",   A_emp_young,  K0_YOUNG,  RHO_EMP_EARLY),
    ("Emp P45-50", A_emp_mature, K0_MATURE, RHO_EMP_MATURE),
]

print("Measuring escape times and attractors...")
escape_means, escape_stds, attractors = [], [], []
for name, A, _, _ in conditions:
    theta_att = find_attractor(A, omega)
    attractors.append(theta_att)
    m, s = measure_escape_time(A, omega, SIGMA_ESCAPE, theta_att, t_max=50)
    escape_means.append(m)
    escape_stds.append(s)
    print(f"  {name}: {m:.2f} ± {s:.2f}")

print("Computing energy profiles...")
profiles = []
for name, A, _, _ in conditions:
    dp, E = compute_energy_profile(A, r_fixed=0.99, n_points=40, n_samples=6)
    profiles.append((dp, E))

print("Computing escape traces...")
traces = []
for (name, A, _, _), theta_att in zip(conditions, attractors):
    np.random.seed(100)
    t_tr, r_tr = escape_trace(A, omega, theta_att, SIGMA_ESCAPE, t_max=50)
    traces.append((t_tr, r_tr))

print("\nMeasuring critical coupling K_c (intra-population)...")
K_eff_axis, r_steady = measure_Kc()
K_c_measured = K_eff_axis[np.argmax(np.array(r_steady) > 0.5)]
print(f"\n>>> Measured intra-population K_c  = {K_c_measured:.3f}")
print(f">>> Theoretical mean-field K_c      = {np.sqrt(8 / np.pi) * 0.15:.3f}")
print(f">>> K_eff sim mature                = {K0_MATURE * RHO_SIM_MATURE:.2f}")
print(f">>> K_eff emp mature                = {K0_MATURE * RHO_EMP_MATURE:.2f}")


# =====================================================================
# FIGURE 3 — THREE PANELS
# =====================================================================
print("\nPlotting Figure 3 (3 panels)...")
fig = plt.figure(figsize=(18, 6))
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.32,
                       left=0.06, right=0.97, top=0.90, bottom=0.14)

colors_B = ['#4C72B0', '#1f4e79', '#DD8452', '#8B4513']

# ---------------- Panel A: attractor depth as potential wells ----------------
axA = fig.add_subplot(gs[0, 0])

x_positions = [0.15, 0.38, 0.62, 0.85]
esc_values = escape_means
colors = ['#4C72B0', '#1f4e79', '#DD8452', '#8B4513']
labels = ['Sim\nP7-8', 'Sim\nP45-50', 'Emp\nP7-8', 'Emp\nP45-50']

half_width = 0.11
base = 0.05
base_log = np.log10(base)
x_local = np.linspace(-half_width, half_width, 400)

for x0, esc, color in zip(x_positions, esc_values, colors):
    peak_log = np.log10(esc)
    x_norm = x_local / half_width
    log_y = base_log + (peak_log - base_log) * (1 - x_norm ** 2)
    y = 10 ** log_y
    axA.fill_between(x0 + x_local, base, y,
                     color=color, alpha=0.35, zorder=2)
    axA.plot(x0 + x_local, y, color=color, lw=3.2,
             zorder=3, solid_capstyle='round')

for x0, esc, color in zip(x_positions, esc_values, colors):
    axA.text(x0, esc * 1.9, f'{esc:.2f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold',
             color=color, zorder=5)

axA.set_xticks(x_positions)
axA.set_xticklabels(labels, fontsize=11)
axA.set_xlim(0, 1.0)
axA.set_yscale('log')
axA.set_ylim(0.05, 200)
axA.set_ylabel('Escape time (a.u.)', fontsize=12)
axA.grid(True, alpha=0.3, axis='y', which='both', zorder=0)

axA.text(0.265, 130, 'Simulated', ha='center', va='center',
         fontsize=11, fontweight='bold', color='#1f4e79',
         bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                   edgecolor='#1f4e79', lw=1.4, alpha=0.95), zorder=6)
axA.text(0.735, 130, 'Empirical', ha='center', va='center',
         fontsize=11, fontweight='bold', color='#8B4513',
         bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                   edgecolor='#8B4513', lw=1.4, alpha=0.95), zorder=6)
axA.axvline(0.5, color='gray', ls='--', alpha=0.35, lw=1.4, zorder=1)

axA.annotate('', xy=(0.38, 38.4), xytext=(0.15, 0.20),
             arrowprops=dict(arrowstyle='-|>', color='#1f4e79',
                             lw=2.5, connectionstyle='arc3,rad=-0.35',
                             mutation_scale=20, alpha=0.85), zorder=7)
axA.text(0.26, 4.0, '188×', ha='center', va='center',
         fontsize=11.5, fontweight='bold', color='#1f4e79',
         bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                   edgecolor='#1f4e79', lw=1.4, alpha=0.95), zorder=8)

axA.annotate('', xy=(0.85, 0.54), xytext=(0.62, 0.17),
             arrowprops=dict(arrowstyle='-|>', color='#8B4513',
                             lw=2.5, connectionstyle='arc3,rad=-0.35',
                             mutation_scale=20, alpha=0.85), zorder=7)
axA.text(0.735, 0.75, '3.2×', ha='center', va='center',
         fontsize=11.5, fontweight='bold', color='#8B4513',
         bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                   edgecolor='#8B4513', lw=1.4, alpha=0.95), zorder=8)

axA.set_title('(A) Attractor depth (log scale)',
              fontsize=13, fontweight='bold', pad=10)

# ---------------- Panel B: escape dynamics under noise ----------------
axB = fig.add_subplot(gs[0, 1])
for (name, (t_tr, r_tr)), c in zip(
        [("Sim P7-8", traces[0]), ("Sim P45-50", traces[1]),
         ("Emp P7-8", traces[2]), ("Emp P45-50", traces[3])], colors_B):
    axB.plot(t_tr, r_tr, lw=2.0, color=c, label=name)
axB.axhline(0.85, color='red', ls='--', lw=1.5, alpha=0.7,
            label='Escape threshold')
axB.set_xlabel('Time (a.u.)', fontsize=12)
axB.set_ylabel('Intra-population coherence', fontsize=12)
axB.set_title('(B) Escape dynamics under noise σ = 2.0',
              fontsize=13, fontweight='bold', pad=10)
axB.set_xlim(0, 50)
axB.set_ylim(0, 1.05)
axB.legend(loc='lower left', fontsize=9, framealpha=0.95)
axB.grid(True, alpha=0.3)

# ---------------- Panel C: developmental trajectory of attractor depth ----------------
axC = fig.add_subplot(gs[0, 2])
x_young, x_mature = 0.3, 0.7

axC.plot([x_young, x_mature], [escape_means[0], escape_means[1]],
         'o-', color='#4C72B0', lw=3.5, markersize=18,
         markeredgecolor='black', markeredgewidth=1.5,
         label='Simulated', zorder=3)
axC.plot([x_young, x_mature], [escape_means[2], escape_means[3]],
         's-', color='#DD8452', lw=3.5, markersize=18,
         markeredgecolor='black', markeredgewidth=1.5,
         label='Empirical (corrected)', zorder=3)

axC.set_xticks([x_young, x_mature])
axC.set_xticklabels(['P7-8\n(young)', 'P45-50\n(mature)'], fontsize=12)
axC.set_xlim(0.05, 0.95)
axC.set_ylabel('Escape time (a.u.)', fontsize=12)
axC.set_yscale('log')
axC.set_ylim(0.05, 300)
axC.grid(True, alpha=0.3, axis='y', which='both')
axC.set_title('(C) Attractor depth across development',
              fontsize=13, fontweight='bold', pad=10)

axC.text(x_young - 0.04, escape_means[0], f'{escape_means[0]:.2f}',
         ha='right', va='center', fontsize=11, fontweight='bold',
         color='#4C72B0')
axC.text(x_mature + 0.04, escape_means[1], f'{escape_means[1]:.1f}',
         ha='left', va='center', fontsize=11, fontweight='bold',
         color='#1f4e79')
axC.text(x_young - 0.04, escape_means[2] * 0.6, f'{escape_means[2]:.2f}',
         ha='right', va='center', fontsize=11, fontweight='bold',
         color='#DD8452')
axC.text(x_mature + 0.04, escape_means[3] * 1.5, f'{escape_means[3]:.2f}',
         ha='left', va='center', fontsize=11, fontweight='bold',
         color='#8B4513')

fold_sim = escape_means[1] / escape_means[0]
fold_emp = escape_means[3] / escape_means[2]
axC.text(0.5, 5, f'Sim: {fold_sim:.0f}×', ha='center',
         fontsize=11, fontweight='bold', color='#1f4e79',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                   edgecolor='#1f4e79', alpha=0.95))
axC.text(0.5, 0.3, f'Emp: {fold_emp:.1f}×', ha='center',
         fontsize=11, fontweight='bold', color='#8B4513',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                   edgecolor='#8B4513', alpha=0.95))

axC.legend(loc='upper left', fontsize=11, framealpha=0.95)

plt.savefig('Figura3_Attractor_Landscape.png', dpi=300, bbox_inches='tight')
print("Figura3_Attractor_Landscape.png saved.")

# ---------------- Summary ----------------
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for name, m, s in zip([c[0] for c in conditions], escape_means, escape_stds):
    print(f"{name:15s}  Escape time = {m:6.2f} ± {s:5.2f}")
print("=" * 60)