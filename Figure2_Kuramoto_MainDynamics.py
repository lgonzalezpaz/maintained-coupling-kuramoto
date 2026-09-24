# -*- coding: utf-8 -*-
"""
Figure 2 — Kuramoto model of maintained coupling: quantitative results.

Reproduces:
    - Figura2_Quantitative.png      (main figure, panels A–H)
    - FiguraS1_Parameter_Sweep.png  (supplementary figure S1)

Companion code for:
    Alvarado et al., "Memory as Maintained Coupling: A Kuramoto Model
    Formalization of Synchronization Dynamics in Neural Systems".

Usage (Google Colab):
    !python Figure2_Kuramoto_MainDynamics.py

Dependencies: numpy, scipy, matplotlib, tqdm
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


# =====================================================================
# PARAMETERS
# =====================================================================
N_POP = 100                    # Number of oscillators per population
N_TOTAL = 2 * N_POP            # Total number of oscillators (two populations)
LAMBDA_SPACE = 0.3             # Spatial decay constant (normalized units)
ETA = 0.01                     # Hebbian learning rate
LAMBDA_K = 0.001               # Synaptic decay rate
K_MIN = 0.01                   # Minimum coupling (below: pruned to zero)
K_MAX = 1000.0                 # Maximum coupling (saturation)

# Effective connection densities (corrected values divided by mean decay)
RHO_INITIAL = 0.0637 / 0.745
RHO_FINAL   = 0.0217 / 0.745

# Coupling strengths
K0_INITIAL   = 500.0
K0_FINAL     = 500.0
K_CROSS_MAX  = 3.0             # Peak cross-coupling (learning protocol)
K_CROSS_MULTI = 0.5            # Pairwise inter-population coupling (multi-pop.)

# Structural plasticity
K0_STRUCTURAL   = 500.0
ALPHA_CREATE    = 0.0005       # Synapse creation probability per step
ALPHA_DELETE    = 0.0005       # Synapse elimination probability per step
DENSITY_CEILING = 0.032        # Upper bound on connection density

np.random.seed(42)


# =====================================================================
# UTILITIES
# =====================================================================
def build_fixed_matrix(N, rho, K0, seed=None):
    """Generate a sparse, distance-dependent coupling matrix on a 1-D ring."""
    if seed is not None:
        np.random.seed(seed)
    positions = np.linspace(0, 1, N)
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = min(abs(positions[i] - positions[j]),
                    1 - abs(positions[i] - positions[j]))
            if np.random.rand() < rho * np.exp(-d / LAMBDA_SPACE):
                base = K0 * np.exp(-d / LAMBDA_SPACE)
                var = np.random.lognormal(mean=0, sigma=0.3)
                K[i, j] = base * var
    K = (K + K.T) / 2
    return K


def circular_std(theta):
    """Circular standard deviation of a phase distribution."""
    R = np.abs(np.mean(np.exp(1j * theta)))
    R = max(R, 1e-10)
    return np.sqrt(-2 * np.log(R))


def intra_cluster_width(theta, N_POP):
    """Mean intra-cluster circular std across the two populations."""
    return 0.5 * (circular_std(theta[:N_POP]) + circular_std(theta[N_POP:]))


def coupling_sum(A, theta):
    """Compute sum_j A_ij * sin(theta_j - theta_i) for every i."""
    sin_diff = np.sin(theta[None, :] - theta[:, None])
    return np.sum(A * sin_diff, axis=1)


# =====================================================================
# BASELINE SIMULATION — LEARNING, CONSOLIDATION, FORGETTING
# =====================================================================
def run_basic_simulation():
    print("Running baseline simulation (2 populations)...")
    N = N_TOTAL
    t_span = (0, 150)
    t_eval = np.linspace(0, 150, 3000)

    omega = np.concatenate([-0.8 + 0.15 * np.random.randn(N_POP),
                             0.8 + 0.15 * np.random.randn(N_POP)])
    theta0 = np.random.uniform(0, 2 * np.pi, N)

    A_base = build_fixed_matrix(N, RHO_FINAL, K0_FINAL, seed=123)
    A_base[:N_POP, N_POP:] = 0
    A_base[N_POP:, :N_POP] = 0

    # Piecewise cross-coupling protocol: baseline / learning / consolidation / forgetting
    K_cross = np.zeros_like(t_eval)
    for idx, t in enumerate(t_eval):
        if t < 30:
            K_cross[idx] = 0
        elif t < 70:
            K_cross[idx] = K_CROSS_MAX * (t - 30) / 40
        elif t < 100:
            K_cross[idx] = K_CROSS_MAX
        else:
            K_cross[idx] = K_CROSS_MAX * max(0, 1 - (t - 100) / 50)

    def ode(t, theta):
        kc = np.interp(t, t_eval, K_cross)
        cross = np.zeros((N, N))
        cross[:N_POP, N_POP:] = 1.0
        cross[N_POP:, :N_POP] = 1.0
        A_total = A_base + kc * cross
        return omega + (1.0 / N_POP) * coupling_sum(A_total, theta)

    sol = solve_ivp(ode, t_span, theta0, t_eval=t_eval,
                    method='RK45', rtol=1e-6)
    r_global = np.abs(np.mean(np.exp(1j * sol.y), axis=0))

    # Phase difference between population centroids (wrapped to [-pi, pi])
    phi1 = np.angle(np.mean(np.exp(1j * sol.y[:N_POP, :]), axis=0))
    phi2 = np.angle(np.mean(np.exp(1j * sol.y[N_POP:, :]), axis=0))
    delta_phi = phi1 - phi2
    delta_phi = np.mod(delta_phi + np.pi, 2 * np.pi) - np.pi

    def circ_mean(x):
        return np.angle(np.mean(np.exp(1j * x)))

    print(f"Delta-phi baseline      (0-30):    {circ_mean(delta_phi[:600]):.3f} rad")
    print(f"Delta-phi consolidation (70-100):  {circ_mean(delta_phi[1400:2000]):.3f} rad")
    print(f"Delta-phi forgetting    (120-150): {circ_mean(delta_phi[2600:]):.3f} rad")

    return t_eval, K_cross, r_global


# =====================================================================
# DEVELOPMENTAL PRUNING SIMULATION (CA3)
# =====================================================================
def run_development_simulation():
    print("Running CA3 developmental pruning simulation...")
    N = N_TOTAL
    t_span = (0, 100)
    t_eval = np.linspace(0, 100, 2000)

    omega = np.concatenate([-0.5 + 0.1 * np.random.randn(N_POP),
                             0.5 + 0.1 * np.random.randn(N_POP)])
    theta0 = np.random.uniform(0, 2 * np.pi, N)

    A_init  = build_fixed_matrix(N, RHO_INITIAL, K0_INITIAL, seed=456)
    A_final = build_fixed_matrix(N, RHO_FINAL,   K0_FINAL,   seed=789)

    def ode(t, theta):
        frac = min(t / 100, 1.0)
        A = (1 - frac) * A_init + frac * A_final
        return omega + (1.0 / N_POP) * coupling_sum(A, theta)

    sol = solve_ivp(ode, t_span, theta0, t_eval=t_eval,
                    method='RK45', rtol=1e-6)

    density_evol, cc_evol = [], []
    for t in t_eval:
        frac = min(t / 100, 1.0)
        A = (1 - frac) * A_init + frac * A_final
        density_evol.append(np.count_nonzero(A) / (N * N))
        if np.count_nonzero(A) > 0:
            cc_evol.append(np.mean([np.mean(A[i, :] > 0) for i in range(N)]))
        else:
            cc_evol.append(0)
    return t_eval, density_evol, cc_evol


# =====================================================================
# MULTI-POPULATION SIMULATION — EMERGENCE OF GLOBAL SYNCHRONY
# =====================================================================
def run_multi_attractor_simulation(M_max=5, K_multi=K_CROSS_MULTI,
                                   omega_range=(-1.0, 1.0)):
    """
    Emergence of global synchrony in ensembles of M coupled populations.

    - Frequency centers are distributed within a FIXED interval, so the
      total frequency spread does not grow with M.
    - r_global is averaged over the last 50 a.u. of a 200-a.u. integration
      to remove transient oscillations in the partially coherent regime.
    """
    r_globals, r_within_list, k_effs = [], [], []

    for m in range(2, M_max + 1):
        centers = np.linspace(omega_range[0], omega_range[1], m)
        N = m * N_POP

        np.random.seed(42)
        omega = np.concatenate([c + 0.1 * np.random.randn(N_POP) for c in centers])
        theta0 = np.random.uniform(0, 2 * np.pi, N)

        # Intra-population blocks
        A = np.zeros((N, N))
        for p in range(m):
            idx = slice(p * N_POP, (p + 1) * N_POP)
            A[idx, idx] = build_fixed_matrix(N_POP, RHO_FINAL, K0_FINAL,
                                             seed=111 + p)

        # Inter-population uniform coupling
        for p in range(m):
            for q in range(m):
                if p != q:
                    A[p * N_POP:(p + 1) * N_POP,
                      q * N_POP:(q + 1) * N_POP] = K_multi

        def ode(t, theta, A=A):
            return omega + (1.0 / N_POP) * coupling_sum(A, theta)

        sol = solve_ivp(ode, (0, 200), theta0,
                        t_eval=np.linspace(0, 200, 2000),
                        method='RK45', rtol=1e-6)

        theta_window = sol.y[:, -500:]
        r_t = np.abs(np.mean(np.exp(1j * theta_window), axis=0))
        r_m = float(np.mean(r_t))

        r_in = np.mean([
            np.abs(np.mean(np.exp(1j * theta_window[p * N_POP:(p + 1) * N_POP, :]),
                           axis=0)).mean()
            for p in range(m)
        ])

        r_globals.append(r_m)
        r_within_list.append(float(r_in))
        k_effs.append((m - 1) * K_multi)

        print(f"  M={m}: r_global={r_m:.3f}, r_within={r_in:.3f}, "
              f"K_eff={(m - 1) * K_multi:.2f}")

    return r_globals, r_within_list, k_effs


# =====================================================================
# STRUCTURAL PLASTICITY — SYNAPTIC CREATION AND ELIMINATION
# =====================================================================
def run_structural_plasticity():
    print("Running structural plasticity simulation...")
    N = N_TOTAL
    A_base = build_fixed_matrix(N, RHO_FINAL, K0_STRUCTURAL, seed=111)
    density_base = np.count_nonzero(A_base) / (N * N)

    omega = np.concatenate([-0.8 + 0.15 * np.random.randn(N_POP),
                             0.8 + 0.15 * np.random.randn(N_POP)])
    theta = np.random.uniform(0, 2 * np.pi, N)
    K = A_base.copy()
    n_steps = 100
    density_hist = [density_base]
    THR_CREATE, THR_DELETE = 0.7, 0.3

    # Warm-up
    for _ in range(200):
        theta = (theta + 0.01 * (omega +
                (1.0 / N_POP) * coupling_sum(K, theta))) % (2 * np.pi)

    for step in range(n_steps):
        theta = (theta + 0.01 * (omega +
                (1.0 / N_POP) * coupling_sum(K, theta))) % (2 * np.pi)
        corr = np.cos(theta[None, :] - theta[:, None])
        current_density = np.count_nonzero(K) / (N * N)

        # Create new synapses between coherent oscillators
        if current_density < DENSITY_CEILING:
            eligible = (K == 0) & (corr > THR_CREATE)
            n_elig = int(np.sum(eligible))
            if n_elig > 0:
                n_create = np.random.binomial(n_elig, ALPHA_CREATE)
                if n_create > 0:
                    idx = np.where(eligible)
                    chosen = np.random.choice(len(idx[0]), n_create, replace=False)
                    K[idx[0][chosen], idx[1][chosen]] = 0.5 * np.random.rand(n_create)

        # Eliminate incoherent connections
        elig_del = (K > 0) & (corr < THR_DELETE)
        n_elig_del = int(np.sum(elig_del))
        if n_elig_del > 0:
            n_del = np.random.binomial(n_elig_del, ALPHA_DELETE)
            if n_del > 0:
                idx = np.where(elig_del)
                chosen = np.random.choice(len(idx[0]), n_del, replace=False)
                K[idx[0][chosen], idx[1][chosen]] = 0

        # Hebbian weight update
        mask_now = K > 0
        delta_K = np.zeros_like(K)
        delta_K[mask_now] = ETA * corr[mask_now] - LAMBDA_K * K[mask_now]
        K = K + delta_K
        K = np.clip(K, 0, K_MAX)
        K[K < K_MIN] = 0
        K = (K + K.T) / 2

        density_hist.append(np.count_nonzero(K) / (N * N))

    return np.arange(n_steps + 1), density_hist


# =====================================================================
# STOCHASTIC NOISE — ROBUSTNESS OF CONSOLIDATED ATTRACTORS
# =====================================================================
def run_noise_analysis():
    print("Running stochastic noise analysis...")
    N = N_TOTAL
    sigma_vals = np.linspace(0, 1.0, 11)
    n_replicas = 20
    escape_rates, widths = [], []

    omega = np.concatenate([-0.8 + 0.15 * np.random.randn(N_POP),
                             0.8 + 0.15 * np.random.randn(N_POP)])
    A = build_fixed_matrix(N, RHO_FINAL, K0_FINAL, seed=222)
    A[:N_POP, N_POP:] = 0
    A[N_POP:, :N_POP] = 0

    for sigma in tqdm(sigma_vals):
        escapes, widths_rep = 0, []
        for rep in range(n_replicas):
            theta = np.random.uniform(0, 2 * np.pi, N)
            dt, n_steps = 0.002, 5000
            r_hist = []
            for _ in range(n_steps):
                dtheta = omega + (1.0 / N_POP) * coupling_sum(A, theta)
                dtheta = dtheta * dt + sigma * np.sqrt(dt) * np.random.randn(N)
                theta = (theta + dtheta) % (2 * np.pi)
                r1 = np.abs(np.mean(np.exp(1j * theta[:N_POP])))
                r2 = np.abs(np.mean(np.exp(1j * theta[N_POP:])))
                r_hist.append(min(r1, r2))
            if np.mean(r_hist[-100:]) < 0.85:
                escapes += 1
            widths_rep.append(intra_cluster_width(theta, N_POP))
        escape_rates.append(escapes / n_replicas)
        widths.append(np.mean(widths_rep))
    return sigma_vals, escape_rates, widths


# =====================================================================
# PARAMETER SENSITIVITY SWEEP
# =====================================================================
def run_parameter_sweep():
    print("Running parameter sensitivity sweep...")
    N = N_TOTAL
    eta_vals    = np.linspace(0.005, 0.020, 7)
    lambda_vals = np.linspace(0.0005, 0.002, 7)
    r_cons_matrix = np.zeros((len(eta_vals), len(lambda_vals)))

    omega = np.concatenate([-0.8 + 0.15 * np.random.randn(N_POP),
                             0.8 + 0.15 * np.random.randn(N_POP)])
    A = build_fixed_matrix(N, RHO_FINAL, K0_FINAL, seed=333)

    for i, eta in enumerate(tqdm(eta_vals)):
        for j, lam in enumerate(lambda_vals):
            K = A.copy()
            theta = np.random.uniform(0, 2 * np.pi, N)
            for _ in range(50):
                theta = (theta + 0.01 * (omega +
                        (1.0 / N_POP) * coupling_sum(K, theta))) % (2 * np.pi)
                corr = np.cos(theta[None, :] - theta[:, None])
                mask_now = K > 0
                delta_K = np.zeros_like(K)
                delta_K[mask_now] = eta * corr[mask_now] - lam * K[mask_now]
                K = K + delta_K
                K = np.clip(K, 0, K_MAX)
                K[K < K_MIN] = 0
            r_cons_matrix[i, j] = np.abs(np.mean(np.exp(1j * theta)))
    return eta_vals, lambda_vals, r_cons_matrix


# =====================================================================
# SUPPLEMENTARY FIGURE S1 — PARAMETER SENSITIVITY HEATMAP
# =====================================================================
def generate_supplementary_figure(eta_vals, lambda_vals, r_sweep):
    """Supplementary Figure S1: end-of-sweep coherence landscape."""
    print("\nGenerating Supplementary Figure S1...")

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.subplots_adjust(left=0.15, right=0.92, top=0.92, bottom=0.12)

    vmin, vmax = r_sweep.min(), r_sweep.max()
    im = ax.imshow(r_sweep,
                   extent=[lambda_vals[0], lambda_vals[-1],
                           eta_vals[0], eta_vals[-1]],
                   origin='lower', cmap='RdYlGn',
                   vmin=vmin, vmax=vmax, aspect='auto')

    X, Y = np.meshgrid(lambda_vals, eta_vals)
    mid_level = (vmin + vmax) / 2
    ax.contour(X, Y, r_sweep, levels=[mid_level],
               colors='black', linewidths=1.5, linestyles='--')

    ax.set_xlabel(r'$\lambda_K$ (decay rate)', fontsize=12, labelpad=10)
    ax.set_ylabel(r'$\eta$ (learning rate)', fontsize=12, labelpad=10)
    ax.set_title('(S1) Parameter sensitivity of end-of-sweep coherence',
                 fontsize=13, fontweight='bold', pad=12)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'$r_{cons}$', fontsize=11)

    plt.savefig('FiguraS1_Parameter_Sweep.png', dpi=300, bbox_inches='tight')
    print("  Saved: FiguraS1_Parameter_Sweep.png")
    plt.show()


# =====================================================================
# MAIN FIGURE 2 — EIGHT PANELS
# =====================================================================
def generate_figures():
    print("Running simulations...")
    t, K_cross, r_global = run_basic_simulation()
    t_dev, density_evol, cc_evol = run_development_simulation()
    r_globals_M, r_within_M, k_effs_M = run_multi_attractor_simulation(5)
    t_struct, density_struct = run_structural_plasticity()
    sigma_vals, escape_rates, widths = run_noise_analysis()
    eta_vals, lambda_vals, r_sweep = run_parameter_sweep()

    fig, axes = plt.subplots(3, 4, figsize=(22, 17))
    fig.subplots_adjust(left=0.055, right=0.945, top=0.945, bottom=0.055,
                        hspace=0.62, wspace=0.42)

    # ---------------- Panel A: cross-coupling protocol ----------------
    ax1 = axes[0, 0]
    ax1.plot(t, K_cross, 'b-', lw=2)
    ax1.set_xlabel('Time (a.u.)', fontsize=11, labelpad=10)
    ax1.set_ylabel('$K_{cross}$', fontsize=13, labelpad=10)
    ax1.set_title('(A) Protocol', fontsize=13, fontweight='bold', pad=12)
    ax1.grid(True, alpha=0.3)

    # ---------------- Panel B: global synchrony ----------------
    ax2 = axes[0, 1]
    ax2.axvspan(0, 30, color='#E3F2FD', alpha=0.30, zorder=0)
    ax2.axvspan(30, 70, color='#FFF3E0', alpha=0.30, zorder=0)
    ax2.axvspan(70, 100, color='#E8F5E9', alpha=0.30, zorder=0)
    ax2.axvspan(100, 150, color='#FFEBEE', alpha=0.30, zorder=0)
    ax2.plot(t, r_global, color='purple', lw=2.5, zorder=4, label='$r_{global}$')
    ax2.axhline(0.9, color='gray', ls='--', alpha=0.65, lw=1.5,
                zorder=2, label='Threshold (0.9)')
    y_max, y_min = float(np.max(r_global)), float(np.min(r_global))
    span = max(y_max - y_min, 0.05)
    label_y = y_max + 0.08 * span
    for x, lbl, col in [(15, 'Baseline', '#1565C0'),
                        (50, 'Learning', '#E65100'),
                        (85, 'Consolidation', '#2E7D32'),
                        (125, 'Forgetting', '#C62828')]:
        ax2.text(x, label_y, lbl, ha='center', fontsize=8,
                 color=col, fontweight='bold')
    ax2.set_xlabel('Time (a.u.)', fontsize=11, labelpad=10)
    ax2.set_ylabel('$r_{global}$', fontsize=13, labelpad=10)
    ax2.set_title('(B) Global synchrony', fontsize=13, fontweight='bold', pad=12)
    ax2.legend(loc='lower left', fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 150)
    ax2.set_ylim(max(0, y_min - 0.08 * span), min(1.10, y_max + 0.15 * span))

    # ---------------- Panel C: CA3 pruning ----------------
    ax3 = axes[0, 2]
    density_arr = np.array(density_evol)
    ax3.plot(t_dev, density_arr, '-', color='#1565C0', lw=2.8, label='Model')
    emp_days = np.array([0, 25, 100])
    emp_vals = np.array([0.0637, 0.0280, 0.0217])
    emp_err  = np.array([0.0128, 0.0081, 0.0065])
    ax3.errorbar(emp_days, emp_vals, yerr=emp_err, fmt='D', color='#C62828',
                 ecolor='#C62828', capsize=8, markersize=11,
                 label='Empirical', zorder=5)

    def exp_decay(x, a, tau):
        return a * np.exp(-x / tau)

    try:
        popt, _ = curve_fit(exp_decay, emp_days, emp_vals,
                            p0=[0.07, 30], maxfev=5000)
        x_fit = np.linspace(0, 100, 300)
        ax3.plot(x_fit, exp_decay(x_fit, *popt), '--', color='#C62828',
                 lw=2.0, alpha=0.75, label=f'Fit ($\\tau$={popt[1]:.1f} d)')
        y_pred = exp_decay(emp_days, *popt)
        r_sq = 1 - np.sum((emp_vals - y_pred) ** 2) / np.sum(
            (emp_vals - emp_vals.mean()) ** 2)
    except Exception:
        r_sq = np.nan

    ax3.text(0.5, 0.60,
             f'$R^2$ = {r_sq:.3f}\nSparsification: {emp_vals[0] / emp_vals[-1]:.2f}×',
             transform=ax3.transAxes, ha='center', va='top', fontsize=9.5,
             color='#C62828', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.45', facecolor='white',
                       edgecolor='#C62828', linewidth=1.4, alpha=0.95))
    ax3.set_xlabel('Postnatal days', fontsize=11, labelpad=10)
    ax3.set_ylabel('Effective density', fontsize=11, labelpad=10)
    ax3.set_title('(C) CA3 pruning', fontsize=13, fontweight='bold', pad=12)
    ax3.legend(loc='upper right', fontsize=7.5, framealpha=0.92)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 0.11)
    ax3.set_xlim(-5, 105)

    # ---------------- Panel D: local structure ----------------
    ax4 = axes[0, 3]
    ax4.plot(t_dev, np.array(cc_evol), color='#8c564b', lw=2.5, label='Model CC')
    ax4.axhspan(0.02, 0.08, color='gray', alpha=0.18, label='Random range')
    ax4.axhline(0.15, color='#2ca02c', ls='--', alpha=0.65, lw=1.8,
                label='Structured threshold')
    ax4.set_xlabel('Postnatal days', fontsize=11, labelpad=10)
    ax4.set_ylabel('Clustering coefficient', fontsize=11, labelpad=10)
    ax4.set_title('(D) Local structure', fontsize=13, fontweight='bold', pad=12)
    ax4.legend(loc='upper right', fontsize=8, framealpha=0.85)
    ax4.grid(True, alpha=0.3)

    # ---------------- Panel E: emergence of global synchrony ----------------
    ax5 = axes[1, 0]
    M_values = np.arange(2, 6)

    ax5.plot(M_values, r_globals_M, 'o-', color='#1565C0',
             lw=2.8, markersize=14, markeredgecolor='white',
             markeredgewidth=1.8, zorder=4, label='$r_{global}$')
    ax5.plot(M_values, r_within_M, 's--', color='#E65100',
             lw=2.0, markersize=10, alpha=0.8, zorder=3,
             label='$r_{within}$ (mean)')
    ax5.axhline(0.85, color='gray', ls='--', alpha=0.65, lw=1.5,
                label='Coherence threshold (0.85)')

    for mv, rv in zip(M_values, r_globals_M):
        ax5.text(mv, rv + 0.035, f'{rv:.2f}', ha='center',
                 fontsize=9.5, fontweight='bold', color='#1565C0')

    ax5b = ax5.twiny()
    ax5b.set_xlim(ax5.get_xlim())
    ax5b.set_xticks(M_values)
    ax5b.set_xticklabels([f'{k:.1f}' for k in k_effs_M], fontsize=9)
    ax5b.set_xlabel('$K_{eff} = (M-1) \\cdot K_{multi}$', fontsize=10, labelpad=8)

    ax5.set_xlabel('Number of coupled populations M', fontsize=11, labelpad=10)
    ax5.set_ylabel('$r$', fontsize=13, labelpad=10)
    ax5.set_title('(E) Emergence of global synchrony',
                  fontsize=13, fontweight='bold', pad=12)
    ax5.legend(loc='center right', fontsize=8.5, framealpha=0.9)
    ax5.grid(True, alpha=0.3)
    ax5.set_xticks([2, 3, 4, 5])
    ax5.set_ylim(0.3, 1.10)
    ax5.set_xlim(1.7, 5.3)

    # ---------------- Panel F: synaptic remodeling ----------------
    ax6 = axes[1, 1]
    density_struct_arr = np.array(density_struct)
    baseline = density_struct_arr[0]
    ax6.plot(t_struct, density_struct_arr, color='#E65100', lw=2.8,
             zorder=4, label='Connection density')
    ax6.axhline(baseline, color='#37474F', ls='--', lw=1.6, zorder=2,
                label=f'Baseline ({baseline * 100:.2f}%)')
    peak_idx = int(np.argmax(density_struct_arr))
    peak_val = density_struct_arr[peak_idx]
    peak_pct = (peak_val - baseline) / baseline * 100
    ax6.plot(t_struct[peak_idx], peak_val, marker='*', color='darkred',
             markersize=22, markeredgecolor='white', markeredgewidth=1.5,
             zorder=5, label=f'Peak (+{peak_pct:.0f}%)')
    ax6.annotate(f'Peak: +{peak_pct:.0f}%',
                 xy=(t_struct[peak_idx], peak_val),
                 xytext=(0.35, 0.72),
                 textcoords='axes fraction',
                 fontsize=10, color='darkred', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5),
                 bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                           edgecolor='darkred', alpha=0.95),
                 zorder=6)
    ax6.set_xlabel('Learning steps', fontsize=11, labelpad=10)
    ax6.set_ylabel('Connection density', fontsize=11, labelpad=10)
    ax6.set_title('(F) Synaptic remodeling', fontsize=13, fontweight='bold', pad=12)
    ax6.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(0, 100)

    # ---------------- Panel G: stability under noise ----------------
    ax7 = axes[1, 2]
    sigma_arr = np.array(sigma_vals)
    escape_arr = np.array(escape_rates)
    success_arr = 1.0 - escape_arr
    ax7.bar(sigma_arr, success_arr, width=0.028, color='#66BB6A',
            edgecolor='#2E7D32', linewidth=1.0, label='Stable (r ≥ 0.85)')
    ax7.bar(sigma_arr, escape_arr, width=0.028, bottom=success_arr,
            color='#EF5350', edgecolor='#C62828', linewidth=1.0,
            label='Escaped (r < 0.85)')
    ax7.axhline(0.95, color='black', ls='--', alpha=0.75, lw=1.8,
                label='95% threshold')
    idx_esc = (np.argmax(escape_arr > 0.05) if np.any(escape_arr > 0.05)
               else len(sigma_arr) - 1)
    ax7.axvline(sigma_arr[idx_esc] + 0.014, color='darkred', ls=':', lw=2.2)
    ax7.text(sigma_arr[idx_esc] + 0.05, 0.55,
             f'$\\sigma_c \\approx {sigma_arr[idx_esc]:.2f}$',
             fontsize=10, color='darkred', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                       edgecolor='darkred', linewidth=1.3))
    ax7.set_xlabel('Noise intensity σ', fontsize=11, labelpad=10)
    ax7.set_ylabel('Fraction of replicates', fontsize=11, labelpad=10)
    ax7.set_title('(G) Stability under noise', fontsize=13,
                  fontweight='bold', pad=12)
    ax7.legend(loc='lower right', fontsize=7.5, framealpha=0.9)
    ax7.grid(True, alpha=0.3, axis='y')
    ax7.set_ylim(0, 1.12)

    # ---------------- Panel H: attractor deformation ----------------
    ax8 = axes[1, 3]
    ax8.plot(sigma_vals, widths, 'green', marker='s', lw=2)
    ax8.set_xlabel('Noise intensity σ', fontsize=11, labelpad=10)
    ax8.set_ylabel('Attractor width (rad)', fontsize=11, labelpad=10)
    ax8.set_title('(H) Attractor deformation', fontsize=13,
                  fontweight='bold', pad=12)
    ax8.grid(True, alpha=0.3)

    plt.savefig('Figura2_Quantitative.png', dpi=300, bbox_inches='tight')
    print("Figura2_Quantitative.png saved.")
    plt.show()

    # ---------------- Summary of numerical results ----------------
    print("\n" + "=" * 60)
    print("QUANTITATIVE RESULTS (FINAL VALUES)")
    print("=" * 60)
    print(f"Baseline r_global:      {np.mean(r_global[:600]):.3f}")
    print(f"Consolidation r_global: {np.mean(r_global[1400:2000]):.3f}")
    print(f"Forgetting r_global:    {np.mean(r_global[2600:2800]):.3f}")
    print(f"Initial density:        {density_evol[0] * 100:.2f}%")
    print(f"Final density:          {density_evol[-1] * 100:.2f}%")
    print("r_global vs M:          " + ", ".join("%.3f" % x for x in r_globals_M))
    print("r_within vs M:          " + ", ".join("%.3f" % x for x in r_within_M))
    print("K_eff vs M:             " + ", ".join("%.2f" % x for x in k_effs_M))
    print(f"Width sigma=0:          {widths[0]:.3f} rad")
    print(f"Width sigma=0.5:        {widths[5]:.3f} rad")
    print(f"sigma_c:                {sigma_vals[idx_esc]:.2f}")
    print(f"Plasticity peak:        {peak_pct:.1f}%")
    print("=" * 60)

    generate_supplementary_figure(eta_vals, lambda_vals, r_sweep)


# =====================================================================
# ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    generate_figures()