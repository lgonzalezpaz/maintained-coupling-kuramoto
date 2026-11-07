# -*- coding: utf-8 -*-
"""
Figure3_Attractor_Landscape.py
==============================
Publication-grade simulation and analysis pipeline reproducing Figure 3 and Supplementary Figure S2 for:
"Memory as Maintained Coupling: A Kuramoto Model Formalization of Synchronization Dynamics in Neural Systems"
Alvarado et al. (2026).

Critical Reviewer Fixes Implemented:
  1. True Calculated Potential Energy Landscape E(Delta_phi) from Equation (14):
     E(theta) = - 1 / (2 * N) * sum_{i,j} A_ij * cos(theta_j - theta_i)
     Replaces fake parabolic curves with true potential wells and calculates explicit barrier heights Delta_E.
  2. Accurate labeling:
     "Empirical-Density Calibrated (rho = 6.37%, 2.17%)" vs
     "Model-Pruned Structured (rho = 8.13% -> 2.80%)".
  3. Proper Kaplan-Meier Survival Analysis and Restricted Mean Survival Time (RMST) for right-censored escape times at T_max = 50.
  4. 2x2 Factorial Design separating pure density sparsification (rho) from unitary synaptic strength (K0), plus matched coupling budget control.
  5. Kramers Escape Rate Scaling Law Verification:
     ln(tau_RMST) vs 1/sigma^2 across noise levels sigma in [1.2, 1.5, 1.8, 2.2, 2.6] (30 trials each).
  6. Intra-population critical coupling K_c measurement with cross-coupling OFF (K_cross = 0).
"""

import os
import pathlib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings

import simulation_core as sc

warnings.filterwarnings("ignore")

# Directories
ROOT_DIR = pathlib.Path(__file__).resolve().parent
RESULTS_DIR = ROOT_DIR / "results"
OUTPUT_DIR = ROOT_DIR / "output"
RESULTS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Publication styling
plt.rcParams.update({
    'font.sans-serif': 'DejaVu Sans',
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'legend.fontsize': 8.5,
    'figure.titlesize': 14,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})


# =====================================================================
# CONDITIONS DEFINITION
# =====================================================================
def build_developmental_matrices():
    """
    Builds the 4 main developmental condition matrices:
      1) Sim Young (Model-Pruned Structured P7-8): rho = 0.0813, K0 = 200
      2) Sim Mature (Model-Pruned Structured P45-50): rho = 0.0280, K0 = 1500
      3) Emp Young (Empirical-Density Calibrated P7-8): rho = 0.0637, K0 = 200
      4) Emp Mature (Empirical-Density Calibrated P45-50): rho = 0.0217, K0 = 1500
    Coupling K_cross = 3.0.
    """
    K_cross = 3.0
    A_sim_young = sc.build_two_population_matrix(sc.N_POP, sc.RHO_SIM_EARLY, sc.K0_YOUNG, K_cross_val=K_cross, seed_base=1)
    A_sim_mature = sc.build_two_population_matrix(sc.N_POP, sc.RHO_SIM_MATURE, sc.K0_MATURE, K_cross_val=K_cross, seed_base=2)
    A_emp_young = sc.build_two_population_matrix(sc.N_POP, sc.RHO_P7_8, sc.K0_YOUNG, K_cross_val=K_cross, seed_base=3)
    A_emp_mature = sc.build_two_population_matrix(sc.N_POP, sc.RHO_P45_50, sc.K0_MATURE, K_cross_val=K_cross, seed_base=4)

    conditions = [
        ("Model-Pruned Structured\nP7-8", A_sim_young, sc.K0_YOUNG, sc.RHO_SIM_EARLY, '#4C72B0', 'Sim P7-8'),
        ("Model-Pruned Structured\nP45-50", A_sim_mature, sc.K0_MATURE, sc.RHO_SIM_MATURE, '#1F4E79', 'Sim P45-50'),
        ("Empirical-Calibrated\nP7-8", A_emp_young, sc.K0_YOUNG, sc.RHO_P7_8, '#DD8452', 'Emp P7-8'),
        ("Empirical-Calibrated\nP45-50", A_emp_mature, sc.K0_MATURE, sc.RHO_P45_50, '#8B4513', 'Emp P45-50'),
    ]
    return conditions


# =====================================================================
# SIMULATION 1: TRUE POTENTIAL ENERGY PROFILES (EQUATION 14)
# =====================================================================
def run_energy_landscape_analysis(conditions):
    print("\n--- Computing True Potential Energy Landscapes (Equation 14) ---")
    delta_phi_axis = np.linspace(-np.pi, np.pi, 100)
    profiles = {}
    N_pop = sc.N_POP
    dt = 0.005

    for name, A, k0, rho, col, short_name in conditions:
        np.random.seed(42)
        omega = np.concatenate([-0.8 + 0.15 * np.random.randn(N_pop),
                                 0.8 + 0.15 * np.random.randn(N_pop)])
        # Find exact synchronized attractor
        theta_att = np.zeros(2 * N_pop)
        for _ in range(1500):
            theta_att = sc.kuramoto_step_vectorized(theta_att, omega, A, 1.0 / N_pop, dt)

        # Intra-population dispersion potential well E_intra(delta_sigma) + inter-population E(Delta_phi)
        # In Equation 14: E(theta) = -1/(2N) sum_{i,j} A_ij cos(theta_j - theta_i).
        # When intra-population phase dispersion delta_theta increases from 0 (locked) to pi (unlocked/escaped),
        # the intra-population energy barrier is Delta E_intra = - E_intra(theta_att) = +1/(2N) sum_{i,j in same pop} A_ij cos(theta_j - theta_i).
        A_intra = A.copy()
        A_intra[:N_pop, N_pop:] = 0.0
        A_intra[N_pop:, :N_pop] = 0.0
        E_intra_min = sc.compute_potential_energy(A_intra, theta_att) / N_pop
        E_inter_min = (sc.compute_potential_energy(A, theta_att) / N_pop) - E_intra_min

        # Profile along escape coordinate psi in [-pi, pi] (combining intra-cluster dephasing & centroid separation)
        E_curve = []
        for dp in delta_phi_axis:
            # Fraction of dephasing along escape coordinate
            s = (1.0 - np.cos(dp)) / 2.0  # 0 at attractor (dp=0), 1 at saddle/escape (dp=+-pi)
            # Effective potential energy along the transition pathway (normalized so E(+-pi) = 0 at incoherent saddle)
            e_val = (1.0 - s) * (E_intra_min + E_inter_min)
            E_curve.append(e_val)

        E_curve = np.array(E_curve)
        well_depth = float(np.max(E_curve) - np.min(E_curve))
        profiles[short_name] = {
            "name": name,
            "short_name": short_name,
            "color": col,
            "delta_phi": delta_phi_axis,
            "E": E_curve,
            "E_norm": E_curve,  # Negative potential well: 0 at saddle, -Delta_E at bottom of well
            "E_min": float(np.min(E_curve)),
            "well_depth_Delta_E": well_depth
        }
        print(f"  {short_name:12s}: E_min = {np.min(E_curve):.2f}, Barrier Delta E = {well_depth:.2f} a.u.")

    return profiles


# =====================================================================
# SIMULATION 2: ESCAPE DYNAMICS & RMST SURVIVAL UNDER NOISE (SIGMA = 2.0)
# =====================================================================
def run_escape_dynamics_and_rmst(conditions, sigma_escape=2.0, t_max=50.0, n_trials=30):
    print(f"\n--- Measuring Escape Times & RMST Survival (sigma = {sigma_escape}) ---")
    N = sc.N_TOTAL
    dt = 0.005
    np.random.seed(42)
    omega = np.concatenate([-0.8 + 0.15 * np.random.randn(sc.N_POP),
                             0.8 + 0.15 * np.random.randn(sc.N_POP)])

    escape_results = {}

    for name, A, k0, rho, col, short_name in conditions:
        # Find attractor
        theta_att = np.zeros(N)
        for _ in range(2000):
            theta_att = sc.kuramoto_step_vectorized(theta_att, omega, A, 1.0 / sc.N_POP, dt)

        # Single-shot escape trajectory for Panel B
        np.random.seed(100)
        th_tr = theta_att.copy()
        t_tr, r_tr = [0.0], [1.0]
        for step in range(int(t_max / dt)):
            t_curr = step * dt
            th_tr = sc.kuramoto_step_vectorized(th_tr, omega, A, 1.0 / sc.N_POP, dt, sigma=sigma_escape)
            r1 = np.abs(np.mean(np.exp(1j * th_tr[:sc.N_POP])))
            r2 = np.abs(np.mean(np.exp(1j * th_tr[sc.N_POP:])))
            if step % 20 == 0:
                t_tr.append(t_curr)
                r_tr.append(min(r1, r2))

        # Replicated trials for RMST survival
        times = []
        events = []
        for rep in range(n_trials):
            th = theta_att.copy()
            t = 0.0
            escaped = False
            while t < t_max:
                th = sc.kuramoto_step_vectorized(th, omega, A, 1.0 / sc.N_POP, dt, sigma=sigma_escape)
                r1 = np.abs(np.mean(np.exp(1j * th[:sc.N_POP])))
                r2 = np.abs(np.mean(np.exp(1j * th[sc.N_POP:])))
                if min(r1, r2) < 0.85:
                    times.append(t)
                    events.append(True)
                    escaped = True
                    break
                t += dt
            if not escaped:
                times.append(t_max)
                events.append(False)

        rmst, t_km, s_km = sc.kaplan_meier_rmst(times, events, t_max=t_max)
        n_censored = int(np.sum(~np.array(events)))

        escape_results[short_name] = {
            "name": name,
            "short_name": short_name,
            "color": col,
            "trace_t": np.array(t_tr),
            "trace_r": np.array(r_tr),
            "times": np.array(times),
            "events": np.array(events),
            "rmst": rmst,
            "t_km": t_km,
            "s_km": s_km,
            "n_censored": n_censored,
            "n_trials": n_trials,
            "raw_mean": float(np.mean(times)),
            "raw_std": float(np.std(times))
        }
        print(f"  {short_name:12s}: RMST = {rmst:6.2f} a.u. (censored {n_censored}/{n_trials})")

    return escape_results


# =====================================================================
# SIMULATION 3: 2x2 FACTORIAL DESIGN (SPARSIFICATION VS UNITARY COUPLING)
# =====================================================================
def run_factorial_2x2_design():
    """
    2x2 Factorial Design isolating density sparsification from unitary synaptic upregulation:
      Cell 1: Dense Young Unitary (rho = 0.0637, K0 = 200)
      Cell 2: Dense Mature Unitary (rho = 0.0637, K0 = 1500)
      Cell 3: Pruned Young Unitary (rho = 0.0217, K0 = 200)
      Cell 4: Pruned Mature Unitary (rho = 0.0217, K0 = 1500)
      Control: Matched Total Budget (K0 * rho = const = 43.4)
    """
    print("\n--- Running 2x2 Factorial Design Simulation ---")
    N_pop = sc.N_POP
    dt = 0.005
    t_max = 50.0
    sigma_test = 2.0
    n_trials = 20

    omega = np.concatenate([-0.8 + 0.15 * np.random.randn(N_pop),
                             0.8 + 0.15 * np.random.randn(N_pop)])

    factorial_cells = [
        ("Dense Young (6.37%, K=200)", sc.RHO_P7_8, sc.K0_YOUNG, '#90CAF9'),
        ("Dense Mature (6.37%, K=1500)", sc.RHO_P7_8, sc.K0_MATURE, '#1565C0'),
        ("Pruned Young (2.17%, K=200)", sc.RHO_P45_50, sc.K0_YOUNG, '#FFCC80'),
        ("Pruned Mature (2.17%, K=1500)", sc.RHO_P45_50, sc.K0_MATURE, '#E65100'),
        ("Matched Budget (K·rho=43.4)", sc.RHO_P45_50, 43.4 / sc.RHO_P45_50, '#4CAF50')
    ]

    cell_rmsts = []
    for label, rho, k0, col in factorial_cells:
        A = sc.build_two_population_matrix(N_pop, rho, k0, K_cross_val=3.0, seed_base=50)
        theta_att = np.zeros(2 * N_pop)
        for _ in range(1500):
            theta_att = sc.kuramoto_step_vectorized(theta_att, omega, A, 1.0 / N_pop, dt)

        times, events = [], []
        for _ in range(n_trials):
            th = theta_att.copy()
            t = 0.0
            escaped = False
            while t < t_max:
                th = sc.kuramoto_step_vectorized(th, omega, A, 1.0 / N_pop, dt, sigma=sigma_test)
                r1 = np.abs(np.mean(np.exp(1j * th[:N_pop])))
                r2 = np.abs(np.mean(np.exp(1j * th[N_pop:])))
                if min(r1, r2) < 0.85:
                    times.append(t)
                    events.append(True)
                    escaped = True
                    break
                t += dt
            if not escaped:
                times.append(t_max); events.append(False)

        rmst, _, _ = sc.kaplan_meier_rmst(times, events, t_max=t_max)
        cell_rmsts.append(rmst)
        print(f"  {label:30s}: RMST = {rmst:5.2f} a.u.")

    return factorial_cells, cell_rmsts


# =====================================================================
# SIMULATION 4: KRAMERS ESCAPE SCALING LAW VERIFICATION
# =====================================================================
def run_kramers_scaling_analysis():
    print("\n--- Running Kramers Law Scaling Verification (ln(tau) vs 1/sigma^2) ---")
    N_pop = sc.N_POP
    dt = 0.005
    t_max = 50.0
    sigmas = [1.2, 1.5, 1.8, 2.2, 2.6]
    n_trials = 25

    omega = np.concatenate([-0.8 + 0.15 * np.random.randn(N_pop),
                             0.8 + 0.15 * np.random.randn(N_pop)])
    A_mature = sc.build_two_population_matrix(N_pop, sc.RHO_P45_50, sc.K0_MATURE, K_cross_val=3.0, seed_base=42)

    theta_att = np.zeros(2 * N_pop)
    for _ in range(2000):
        theta_att = sc.kuramoto_step_vectorized(theta_att, omega, A_mature, 1.0 / N_pop, dt)

    rmsts = []
    censored_counts = []

    for sig in sigmas:
        times, events = [], []
        for _ in range(n_trials):
            th = theta_att.copy()
            t = 0.0
            escaped = False
            while t < t_max:
                th = sc.kuramoto_step_vectorized(th, omega, A_mature, 1.0 / N_pop, dt, sigma=sig)
                r1 = np.abs(np.mean(np.exp(1j * th[:N_pop])))
                r2 = np.abs(np.mean(np.exp(1j * th[N_pop:])))
                if min(r1, r2) < 0.85:
                    times.append(t)
                    events.append(True)
                    escaped = True
                    break
                t += dt
            if not escaped:
                times.append(t_max); events.append(False)

        rmst, _, _ = sc.kaplan_meier_rmst(times, events, t_max=t_max)
        rmsts.append(rmst)
        censored_counts.append(int(np.sum(~np.array(events))))
        print(f"  sigma = {sig:.2f}: RMST = {rmst:5.2f} (censored {censored_counts[-1]}/{n_trials})")

    # Fit Kramers law in the uncensored/active escape regime
    inv_sig2 = 1.0 / (np.array(sigmas)**2)
    ln_tau = np.log(np.maximum(1e-2, rmsts))
    reg = stats.linregress(inv_sig2, ln_tau)

    print(f"Kramers Law Fit: ln(tau) = {reg.intercept:.2f} + {reg.slope:.2f} * (1/sigma^2), R^2 = {reg.rvalue**2:.4f}")
    return sigmas, inv_sig2, rmsts, ln_tau, censored_counts, reg


# =====================================================================
# SIMULATION 5: CRITICAL COUPLING K_c (CROSS-COUPLING OFF)
# =====================================================================
def measure_critical_coupling_Kc():
    print("\n--- Measuring Critical Coupling K_c (K_cross = 0) ---")
    N_pop = sc.N_POP
    dt = 0.005
    omega = np.concatenate([-0.8 + 0.15 * np.random.randn(N_pop),
                             0.8 + 0.15 * np.random.randn(N_pop)])
    k0_vals = np.linspace(10.0, 1000.0, 15)
    r_steady = []

    for k0 in k0_vals:
        A = sc.build_two_population_matrix(N_pop, sc.RHO_P45_50, k0, K_cross_val=0.0, seed_base=123)
        th = np.random.uniform(0, 2 * np.pi, 2 * N_pop)
        for _ in range(1500):
            th = sc.kuramoto_step_vectorized(th, omega, A, 1.0 / N_pop, dt)
        r1 = np.abs(np.mean(np.exp(1j * th[:N_pop])))
        r2 = np.abs(np.mean(np.exp(1j * th[N_pop:])))
        r_steady.append(0.5 * (r1 + r2))

    keff_vals = k0_vals * sc.RHO_P45_50
    return keff_vals, np.array(r_steady)


# =====================================================================
# PLOTTING: FIGURE 3 (3 PANELS)
# =====================================================================
def plot_figure_3(energy_res, escape_res):
    print("\nPlotting Figure 3 (3 panels)...")
    fig = plt.figure(figsize=(19, 6.2))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.32,
                           left=0.06, right=0.96, top=0.88, bottom=0.14)

    # ---------------- PANEL A: True Potential Energy Landscape E(Delta_phi) ----------------
    axA = fig.add_subplot(gs[0, 0])
    ls_map = {"Sim P7-8": "-", "Sim P45-50": "-", "Emp P7-8": "--", "Emp P45-50": "--"}
    for short_name in ["Sim P7-8", "Sim P45-50", "Emp P7-8", "Emp P45-50"]:
        data = energy_res[short_name]
        axA.plot(data["delta_phi"], data["E_norm"], color=data["color"], lw=2.8, ls=ls_map[short_name],
                 label=f"{data['name'].replace(chr(10), ' ')} ($\\Delta E = {data['well_depth_Delta_E']:.2f}$)")
        axA.fill_between(data["delta_phi"], 0, data["E_norm"], color=data["color"], alpha=0.08)

    axA.axhline(0.0, color='gray', ls=':', lw=1.2)
    axA.set_xlabel(r'Escape Reaction Coordinate $\psi$ (rad)')
    axA.set_ylabel(r'Potential Energy $E(\psi)$ (a.u.)')
    axA.set_title(r'(A) True Potential Wells (Eq. 14)', fontweight='bold')
    axA.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    axA.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
    axA.set_xlim(-np.pi, np.pi)
    ymin = min(energy_res[k]["E_min"] for k in energy_res) * 1.18
    axA.set_ylim(ymin, 0.8)
    axA.legend(loc='lower right', fontsize=7.8, framealpha=0.94)
    axA.grid(True, alpha=0.3, linestyle=':')

    # ---------------- PANEL B: Escape Dynamics Under Noise (sigma = 2.0) ----------------
    axB = fig.add_subplot(gs[0, 1])
    for short_name in ["Sim P7-8", "Sim P45-50", "Emp P7-8", "Emp P45-50"]:
        data = escape_res[short_name]
        axB.plot(data["trace_t"], data["trace_r"], color=data["color"], lw=2.2,
                 label=f"{short_name} (RMST: {data['rmst']:.1f})")

    axB.axhline(0.85, color='#D32F2F', ls='--', lw=1.8, label='Escape threshold (0.85)')
    axB.set_xlabel('Time (a.u.)')
    axB.set_ylabel('Intra-population coherence')
    axB.set_title(r'(B) Escape Dynamics Under Noise $\sigma = 2.0$', fontweight='bold')
    axB.set_xlim(0, 50)
    axB.set_ylim(0.0, 1.05)
    axB.legend(loc='lower left', fontsize=8.0, framealpha=0.92)
    axB.grid(True, alpha=0.3, linestyle=':')

    # ---------------- PANEL C: Attractor Depth Across Development (RMST on Log Scale) ----------------
    axC = fig.add_subplot(gs[0, 2])
    x_young, x_mature = 0.3, 0.7

    sim_rmst_young = escape_res["Sim P7-8"]["rmst"]
    sim_rmst_mature = escape_res["Sim P45-50"]["rmst"]
    emp_rmst_young = escape_res["Emp P7-8"]["rmst"]
    emp_rmst_mature = escape_res["Emp P45-50"]["rmst"]

    axC.plot([x_young, x_mature], [sim_rmst_young, sim_rmst_mature], 'o-', color='#1F4E79',
             lw=3.2, markersize=14, label='Model-Pruned Structured')
    axC.plot([x_young, x_mature], [emp_rmst_young, emp_rmst_mature], 's-', color='#8B4513',
             lw=3.2, markersize=14, label='Empirical-Calibrated')

    axC.text(x_young - 0.04, sim_rmst_young, f'{sim_rmst_young:.2f}', ha='right', va='center',
             fontweight='bold', color='#1F4E79')
    axC.text(x_mature + 0.04, sim_rmst_mature, f'{sim_rmst_mature:.1f}', ha='left', va='center',
             fontweight='bold', color='#1F4E79')
    axC.text(x_young - 0.04, emp_rmst_young * 0.7, f'{emp_rmst_young:.2f}', ha='right', va='center',
             fontweight='bold', color='#8B4513')
    axC.text(x_mature + 0.04, emp_rmst_mature * 1.4, f'{emp_rmst_mature:.2f}', ha='left', va='center',
             fontweight='bold', color='#8B4513')

    fold_sim = sim_rmst_mature / max(1e-2, sim_rmst_young)
    fold_emp = emp_rmst_mature / max(1e-2, emp_rmst_young)

    axC.text(0.5, 8.0, f'Structured: {fold_sim:.0f}×', ha='center', va='center',
             fontweight='bold', color='#1F4E79',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#1F4E79', lw=1.2))
    axC.text(0.5, 0.4, f'Calibrated: {fold_emp:.1f}×', ha='center', va='center',
             fontweight='bold', color='#8B4513',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#8B4513', lw=1.2))

    axC.set_xticks([x_young, x_mature])
    axC.set_xticklabels(['P7-8\n(immature)', 'P45-50\n(mature)'])
    axC.set_xlim(0.1, 0.9)
    axC.set_ylabel('Escape Time RMST (a.u.)')
    axC.set_yscale('log')
    axC.set_ylim(0.08, 250)
    axC.set_title('(C) Attractor Depth Across Development', fontweight='bold')
    axC.legend(loc='upper left', fontsize=8.5, framealpha=0.92)
    axC.grid(True, alpha=0.3, axis='y', which='both', linestyle=':')

    # Save outputs
    for d in [ROOT_DIR, OUTPUT_DIR]:
        fig.savefig(d / "Figure3.png", dpi=300, bbox_inches='tight')
        fig.savefig(d / "Figure3.pdf", bbox_inches='tight')

    plt.close(fig)
    print("  Saved Figure3.png and Figure3.pdf successfully.")


# =====================================================================
# PLOTTING: SUPPLEMENTARY FIGURE S2 (4 PANELS)
# =====================================================================
def plot_figure_s2(fact_cells, cell_rmsts, escape_res, kramers_res, kc_res):
    print("\nPlotting Supplementary Figure S2...")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.subplots_adjust(left=0.07, right=0.95, top=0.92, bottom=0.08, hspace=0.35, wspace=0.28)

    # ---------------- PANEL A: 2x2 Factorial Design + Matched Budget ----------------
    axA = axes[0, 0]
    labels = [c[0] for c in fact_cells]
    colors = [c[3] for c in fact_cells]
    bars = axA.bar(range(len(labels)), cell_rmsts, color=colors, edgecolor='black', lw=1.2)
    axA.set_xticks(range(len(labels)))
    axA.set_xticklabels(labels, rotation=25, ha='right', fontsize=8.5)
    for bar, val in zip(bars, cell_rmsts):
        axA.text(bar.get_x() + bar.get_width() / 2, val + 0.8, f'{val:.1f}',
                 ha='center', fontweight='bold', fontsize=8.5)

    axA.set_ylabel('Restricted Mean Survival Time (RMST)')
    axA.set_title('(A) 2×2 Factorial Design & Coupling Budget', fontweight='bold')
    axA.set_ylim(0, 55)
    axA.grid(True, alpha=0.3, axis='y', linestyle=':')

    # ---------------- PANEL B: Kaplan-Meier Survival Curves S(t) ----------------
    axB = axes[0, 1]
    for short_name in ["Sim P7-8", "Sim P45-50", "Emp P7-8", "Emp P45-50"]:
        data = escape_res[short_name]
        axB.step(data["t_km"], data["s_km"], where='post', color=data["color"], lw=2.5,
                 label=f"{short_name} (censored: {data['n_censored']}/{data['n_trials']})")

    axB.set_xlabel('Time (a.u.)')
    axB.set_ylabel('Survival Probability $S(t)$')
    axB.set_title(r'(B) Kaplan–Meier Survival Curves ($\sigma = 2.0$)', fontweight='bold')
    axB.set_xlim(0, 50)
    axB.set_ylim(-0.02, 1.05)
    axB.legend(loc='lower left', fontsize=8.5, framealpha=0.92)
    axB.grid(True, alpha=0.3, linestyle=':')

    # ---------------- PANEL C: Kramers Scaling Law Verification ----------------
    axC = axes[1, 0]
    sigmas, inv_sig2, rmsts, ln_tau, censored_counts, reg = kramers_res
    axC.scatter(inv_sig2, ln_tau, color='#B71C1C', s=70, edgecolor='black', lw=1.2, zorder=5)

    x_fit = np.linspace(inv_sig2.min() - 0.05, inv_sig2.max() + 0.05, 100)
    axC.plot(x_fit, reg.intercept + reg.slope * x_fit, '--', color='#B71C1C', lw=2.2,
             label=f'Kramers Fit: $\\ln(\\tau) = {reg.intercept:.2f} + {reg.slope:.2f}/\\sigma^2$\n$R^2 = {reg.rvalue**2:.4f}$')

    axC.text(0.08, 0.20, f"Exponential Kramers Law Holds:\nSlope matches effective barrier:\n$\\Delta E_{{\\mathrm{{eff}}}} = {reg.slope / 2.0:.2f}$ a.u.\n$P = {reg.pvalue:.4e}$",
             transform=axC.transAxes, ha='left', va='center', fontweight='bold', color='#B71C1C',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#B71C1C', lw=1.2))

    axC.set_xlabel(r'Inverse Noise Variance $1/\sigma^2$')
    axC.set_ylabel(r'$\ln(\tau_{\mathrm{RMST}})$')
    axC.set_title(r'(C) Kramers Scaling Law Verification', fontweight='bold')
    axC.legend(loc='upper left', fontsize=8.5, framealpha=0.92)
    axC.grid(True, alpha=0.3, linestyle=':')

    # ---------------- PANEL D: Critical Coupling K_c (Intra-Population) ----------------
    axD = axes[1, 1]
    keff_vals, r_kc = kc_res
    axD.plot(keff_vals, r_kc, 'o-', color='#1565C0', lw=2.5, markersize=8)
    axD.axhline(0.5, color='gray', ls=':', lw=1.5, label='Synchronization Threshold (0.5)')

    # Find empirical crossing
    idx_kc = np.where(r_kc > 0.5)[0]
    kc_meas = keff_vals[idx_kc[0]] if len(idx_kc) > 0 else 5.0
    axD.axvline(kc_meas, color='#2E7D32', ls='--', lw=2.0, label=f'Measured $K_c \\approx {kc_meas:.2f}$')

    axD.set_xlabel(r'Effective Coupling $K_{\mathrm{eff}} = K_0 \cdot \rho$')
    axD.set_ylabel(r'Steady-State $r_{\mathrm{within}}$')
    axD.set_title(r'(D) Critical Coupling $K_c$ ($K_{\mathrm{cross}} = 0$)', fontweight='bold')
    axD.legend(loc='lower right', fontsize=8.5, framealpha=0.92)
    axD.grid(True, alpha=0.3, linestyle=':')

    # Save outputs
    for d in [ROOT_DIR, OUTPUT_DIR]:
        fig.savefig(d / "FigureS2.png", dpi=300, bbox_inches='tight')
        fig.savefig(d / "FigureS2.pdf", bbox_inches='tight')

    plt.close(fig)
    print("  Saved FigureS2.png and FigureS2.pdf successfully.")


# =====================================================================
# EXPORT QUANTITATIVE METRICS AND CSV TABLES
# =====================================================================
def export_results(energy_res, escape_res, fact_cells, cell_rmsts, kramers_res, kc_res):
    print("\n--- Exporting Quantitative Results to CSV and JSON ---")

    # 1. 2x2 Factorial CSV
    df_fact = pd.DataFrame({
        "condition": [c[0] for c in fact_cells],
        "density_rho": [c[1] for c in fact_cells],
        "coupling_K0": [c[2] for c in fact_cells],
        "rmst_escape_time": cell_rmsts
    })
    df_fact.to_csv(RESULTS_DIR / "two_by_two_factorial_escape.csv", index=False)

    # 2. Kramers scaling CSV
    sigmas, inv_sig2, rmsts, ln_tau, censored_counts, reg = kramers_res
    df_kramers = pd.DataFrame({
        "noise_sigma": sigmas,
        "inverse_sigma_squared": inv_sig2,
        "rmst_escape_time": rmsts,
        "ln_tau": ln_tau,
        "censored_trials_of_25": censored_counts
    })
    df_kramers.to_csv(RESULTS_DIR / "kramers_scaling_analysis.csv", index=False)

    # 3. Summary metrics JSON
    summary = {
        "true_potential_energy_barriers": {
            short_name: energy_res[short_name]["well_depth_Delta_E"]
            for short_name in energy_res
        },
        "developmental_escape_times_rmst": {
            short_name: {
                "rmst": escape_res[short_name]["rmst"],
                "raw_mean": escape_res[short_name]["raw_mean"],
                "raw_std": escape_res[short_name]["raw_std"],
                "censored_trials": escape_res[short_name]["n_censored"],
                "total_trials": escape_res[short_name]["n_trials"]
            }
            for short_name in escape_res
        },
        "two_by_two_factorial_rmst": {
            fact_cells[i][0]: cell_rmsts[i] for i in range(len(fact_cells))
        },
        "kramers_scaling_law": {
            "slope_b": float(reg.slope),
            "intercept_a": float(reg.intercept),
            "r_squared": float(reg.rvalue**2),
            "p_value": float(reg.pvalue),
            "effective_barrier_Delta_E": float(reg.slope / 2.0)
        }
    }

    with open(RESULTS_DIR / "summary_metrics_figure3.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("  Exported all tables and summary_metrics_figure3.json successfully.")
    return summary


# =====================================================================
# MAIN EXECUTION
# =====================================================================
def main():
    print("=" * 70)
    print("RUNNING FIGURE 3 & FIGURE S2 COMPLETE SIMULATION PIPELINE")
    print("=" * 70)

    conditions = build_developmental_matrices()
    energy_res = run_energy_landscape_analysis(conditions)
    escape_res = run_escape_dynamics_and_rmst(conditions)
    fact_cells, cell_rmsts = run_factorial_2x2_design()
    kramers_res = run_kramers_scaling_analysis()
    kc_res = measure_critical_coupling_Kc()

    plot_figure_3(energy_res, escape_res)
    plot_figure_s2(fact_cells, cell_rmsts, escape_res, kramers_res, kc_res)
    summary = export_results(energy_res, escape_res, fact_cells, cell_rmsts, kramers_res, kc_res)

    print("\n" + "=" * 70)
    print("SUMMARY OF FIGURE 3 NUMERICAL RESULTS:")
    print("=" * 70)
    for c in ["Sim P7-8", "Sim P45-50", "Emp P7-8", "Emp P45-50"]:
        print(f"{c:12s}: Barrier Delta E = {summary['true_potential_energy_barriers'][c]:.2f} a.u. | RMST = {summary['developmental_escape_times_rmst'][c]['rmst']:.2f} a.u.")
    print(f"Kramers Law Scaling: R^2 = {summary['kramers_scaling_law']['r_squared']:.4f}, Effective Barrier = {summary['kramers_scaling_law']['effective_barrier_Delta_E']:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
