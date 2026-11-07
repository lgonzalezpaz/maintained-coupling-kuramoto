# -*- coding: utf-8 -*-
"""
Figure1_Conceptual_Framework.py
===============================
Generates publication-grade Figure 1 for:
"Memory as Maintained Coupling: A Kuramoto Model Formalization of Synchronization Dynamics in Neural Systems"
Alvarado et al. (2026).

Produces:
  - Figure1.png (300 DPI)
  - Figure1.pdf (vector)
  - output/Figure1.png
  - output/Figure1.pdf

Panels:
  (A) Baseline: low cross-synchrony (polar phase distribution)
  (B) Consolidation: high global synchrony (phase-locked attractor)
  (C) Phase locking by cross-coupling trajectory Delta_phi(t)
  (D) Distance-dependent connectivity profile P(d) and K(d)/K0
  (E) Kuramoto coupling function sin(theta_j - theta_i) with attraction/repulsion basins
  (F) Conceptual architecture and formalization principles
"""

import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

# Set publication style
plt.rcParams.update({
    'font.sans-serif': 'DejaVu Sans',
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10.5,
    'ytick.labelsize': 10.5,
    'legend.fontsize': 10,
    'figure.titlesize': 15,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})


def generate_figure1():
    print("Generating Figure 1 (Conceptual Framework)...")
    np.random.seed(42)

    fig = plt.figure(figsize=(19, 11.5))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.28,
                           left=0.06, right=0.96, top=0.92, bottom=0.08)

    # -------------------------------------------------------------
    # PANEL A: Polar Plot - Baseline (Uncoupled / Drifting)
    # -------------------------------------------------------------
    axA = fig.add_subplot(gs[0, 0], projection='polar')
    n_pts = 100
    # Pop 1 centered around 0 rad, Pop 2 centered around 140 deg (~2.44 rad)
    th1_base = np.random.vonmises(0.0, 18.0, n_pts)
    th2_base = np.random.vonmises(2.44, 16.0, n_pts)
    r1_pts = np.ones(n_pts)
    r2_pts = np.ones(n_pts)

    axA.scatter(th1_base, r1_pts, color='#1976D2', s=35, alpha=0.85,
                edgecolors='none', label='Population 1 ($\\omega_1 \\approx -0.8$)')
    axA.scatter(th2_base, r2_pts, color='#D32F2F', s=35, alpha=0.85,
                edgecolors='none', label='Population 2 ($\\omega_2 \\approx +0.8$)')
    axA.set_ylim(0, 1.15)
    axA.set_yticks([])
    axA.set_theta_zero_location('E')
    axA.set_title('(A) Baseline: low cross-synchrony', pad=18, fontweight='bold', fontsize=13)
    axA.grid(True, alpha=0.4, linestyle=':')
    axA.legend(loc='lower center', bbox_to_anchor=(0.5, -0.26), ncol=1, frameon=True,
               fancybox=True, edgecolor='gray', framealpha=0.9, fontsize=9.5)

    # -------------------------------------------------------------
    # PANEL B: Polar Plot - Consolidated (Coupled / Locked)
    # -------------------------------------------------------------
    axB = fig.add_subplot(gs[0, 1], projection='polar')
    # Phase locked with small offset Delta_phi ~ 0.28 rad
    th1_cons = np.random.vonmises(0.14, 25.0, n_pts)
    th2_cons = np.random.vonmises(-0.14, 25.0, n_pts)

    axB.scatter(th1_cons, r1_pts, color='#1976D2', s=35, alpha=0.85,
                edgecolors='none', label='Population 1')
    axB.scatter(th2_cons, r2_pts, color='#D32F2F', s=35, alpha=0.85,
                edgecolors='none', label='Population 2')
    axB.set_ylim(0, 1.15)
    axB.set_yticks([])
    axB.set_theta_zero_location('E')
    axB.set_title('(B) Consolidation: high global synchrony', pad=18, fontweight='bold', fontsize=13)
    axB.grid(True, alpha=0.4, linestyle=':')
    axB.annotate('Locked Attractor\n$\\Delta\\phi \\approx 0.28\\,\\mathrm{rad}$',
                 xy=(0.0, 1.0), xytext=(np.pi / 2, 0.45),
                 arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=2),
                 ha='center', va='center', fontweight='bold', color='#2E7D32',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#2E7D32', lw=1.2))

    # -------------------------------------------------------------
    # PANEL C: Phase Locking Trajectory Delta_phi(t)
    # -------------------------------------------------------------
    axC = fig.add_subplot(gs[0, 2])
    t = np.linspace(0, 150, 600)
    # Baseline: 2.5 rad; Learning (30-70): drops to 0.28 rad; Consolidation (70-100): 0.28 rad; Forgetting (100-150): rises back
    dp = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti <= 30:
            dp[i] = 2.50
        elif ti <= 70:
            s = (ti - 30.0) / 40.0
            dp[i] = 2.50 - (2.50 - 0.28) * (0.5 - 0.5 * np.cos(np.pi * s))
        elif ti <= 100:
            dp[i] = 0.28
        else:
            s = (ti - 100.0) / 50.0
            dp[i] = 0.28 + (2.50 - 0.28) * (0.5 - 0.5 * np.cos(np.pi * s))

    # Add background shading
    axC.axvspan(0, 30, color='#E3F2FD', alpha=0.45, label='Baseline')
    axC.axvspan(30, 70, color='#FFF3E0', alpha=0.45, label='Learning')
    axC.axvspan(70, 100, color='#E8F5E9', alpha=0.45, label='Consolidation')
    axC.axvspan(100, 150, color='#FFEBEE', alpha=0.45, label='Synaptic Decay')

    axC.plot(t, dp, color='#1B5E20', lw=3.2, label='$\\Delta\\phi$ (phase lag)')
    axC.axhline(0.28, color='#2E7D32', ls=':', lw=1.5, alpha=0.7)
    axC.set_xlabel('Time (a.u.)', labelpad=8)
    axC.set_ylabel('Phase difference $\\Delta\\phi$ (rad)', labelpad=8)
    axC.set_title('(C) Phase locking by cross-coupling', fontweight='bold', pad=12)
    axC.set_xlim(0, 150)
    axC.set_ylim(-0.1, 2.9)
    axC.grid(True, alpha=0.35, linestyle=':')

    # Labels for protocol zones
    axC.text(15, 2.65, 'Baseline\n(drifting)', ha='center', va='center', fontsize=8.5, color='#1565C0', fontweight='bold')
    axC.text(50, 2.65, 'Learning\n(drive on)', ha='center', va='center', fontsize=8.5, color='#E65100', fontweight='bold')
    axC.text(85, 2.65, 'Consolidation\n(maintained)', ha='center', va='center', fontsize=8.5, color='#2E7D32', fontweight='bold')
    axC.text(125, 2.65, 'Decay\n(relaxing)', ha='center', va='center', fontsize=8.5, color='#C62828', fontweight='bold')
    axC.legend(loc='lower left', fontsize=9.5, framealpha=0.92)

    # -------------------------------------------------------------
    # PANEL D: Distance-Dependent Connectivity Profile
    # -------------------------------------------------------------
    axD = fig.add_subplot(gs[1, 0])
    d_axis = np.linspace(0, 1, 300)
    lambda_space = 0.3
    p_d = np.exp(-d_axis / lambda_space)

    axD.plot(d_axis, p_d, color='#0D47A1', lw=3.0, label='Connection probability $P(d)$')
    axD.plot(d_axis, p_d, color='#D32F2F', lw=2.2, ls='--', label='Normalized coupling $K(d)/K_0$')
    axD.fill_between(d_axis, 0, p_d, color='#1976D2', alpha=0.15)
    axD.axvline(lambda_space, color='gray', ls=':', lw=1.8, label=f'Decay scale $\\lambda = {lambda_space}$')
    axD.text(lambda_space + 0.03, 0.85, f'$\\lambda = {lambda_space}$', fontsize=11, color='#424242', fontweight='bold')

    axD.set_xlabel('Normalized ring distance $d$ (a.u.)', labelpad=8)
    axD.set_ylabel('Probability / Normalized strength', labelpad=8)
    axD.set_title('(D) Distance-dependent connectivity', fontweight='bold', pad=12)
    axD.set_xlim(0, 1)
    axD.set_ylim(0, 1.1)
    axD.grid(True, alpha=0.35, linestyle=':')
    axD.legend(loc='upper right', fontsize=9.5, framealpha=0.92)

    # -------------------------------------------------------------
    # PANEL E: Kuramoto Coupling Function with Basins
    # -------------------------------------------------------------
    axE = fig.add_subplot(gs[1, 1])
    dth_axis = np.linspace(-np.pi, np.pi, 400)
    sin_coupling = np.sin(dth_axis)

    axE.plot(dth_axis, sin_coupling, color='#4A148C', lw=3.2)
    axE.axhline(0, color='gray', ls='-', lw=1.0)
    axE.axvline(0, color='gray', ls=':', lw=1.2)

    # Fill attraction (positive) and repulsion (negative)
    axE.fill_between(dth_axis, 0, sin_coupling, where=(dth_axis >= 0),
                     color='#81C784', alpha=0.35, label='Attraction $(\\sin > 0)$')
    axE.fill_between(dth_axis, 0, sin_coupling, where=(dth_axis < 0),
                     color='#EF9A9A', alpha=0.35, label='Repulsion $(\\sin < 0)$')

    axE.set_xlabel('Phase difference $(\\theta_j - \\theta_i)$ (rad)', labelpad=8)
    axE.set_ylabel('Coupling torque $\\sin(\\theta_j - \\theta_i)$', labelpad=8)
    axE.set_title('(E) Kuramoto coupling function', fontweight='bold', pad=12)
    axE.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    axE.set_xticklabels(['$-\\pi$', '$-\\pi/2$', '$0$', '$\\pi/2$', '$\\pi$'])
    axE.set_xlim(-np.pi, np.pi)
    axE.set_ylim(-1.18, 1.18)
    axE.grid(True, alpha=0.35, linestyle=':')
    axE.legend(loc='lower right', fontsize=9.5, framealpha=0.92)

    # -------------------------------------------------------------
    # PANEL F: Conceptual Summary & Formalization Box
    # -------------------------------------------------------------
    axF = fig.add_subplot(gs[1, 2])
    axF.axis('off')

    summary_text = (
        r"$\bf{Formal\ Model\ Architecture}$" + "\n"
        r"• $\bf{Populations}$: $N = 200$ phase oscillators ($N_1 = 100, N_2 = 100$)" + "\n"
        r"  $\omega_1 \sim \mathcal{N}(-0.8, 0.15^2),\ \omega_2 \sim \mathcal{N}(+0.8, 0.15^2)$" + "\n"
        r"• $\bf{Kuramoto\ Dynamics}$:" + "\n"
        r"  $\frac{d\theta_i}{dt} = \omega_i + \frac{1}{N_{\mathrm{pop}}} \sum_{j} A_{ij} \sin(\theta_j - \theta_i) + I_i^{\mathrm{ext}}(t) + \xi_i(t)$" + "\n"
        r"• $\bf{Endogenous\ Hebbian\ Plasticity}$:" + "\n"
        r"  $\frac{dK_{ij}}{dt} = \eta \cos(\theta_j - \theta_i) - \lambda_K K_{ij},\quad K_{ij} \in [0, K_{\max}]$" + "\n"
        r"• $\bf{Structural\ Pruning}$:" + "\n"
        r"  $E(t_2) \subset E(t_1)\ \mathrm{for}\ t_2 > t_1;\ \rho: 6.37\% \to 2.17\%$" + "\n\n"
        r"$\bf{Core\ Theoretical\ Principle}$" + "\n"
        r"$\mathrm{\mathbf{Memory\ is\ not\ a\ passive\ static\ address,\ but\ a}}$" + "\n"
        r"$\mathrm{\mathbf{maintained\ dynamical\ coupling\ topology.}}$" + "\n"
        r"Acquired weights sculpt a deep attractor well in the phase" + "\n"
        r"landscape that sustains phase-locked synchronization" + "\n"
        r"autonomously after external task drive ceases."
    )

    box = FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.04",
                         facecolor="#FAFAFA", edgecolor="#455A64", lw=1.6,
                         alpha=0.95, transform=axF.transAxes)
    axF.add_patch(box)
    axF.text(0.06, 0.92, summary_text, transform=axF.transAxes,
             fontsize=9.8, va='top', ha='left', linespacing=1.35, color='#212121')

    # Save outputs
    root_dir = pathlib.Path(__file__).resolve().parent
    output_dir = root_dir / "output"
    output_dir.mkdir(exist_ok=True)

    fig.savefig(root_dir / "Figure1.png", dpi=300, bbox_inches='tight')
    fig.savefig(root_dir / "Figure1.pdf", bbox_inches='tight')
    fig.savefig(output_dir / "Figure1.png", dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / "Figure1.pdf", bbox_inches='tight')

    plt.close(fig)
    print(f"  Successfully saved Figure1 to {root_dir}")


if __name__ == "__main__":
    generate_figure1()
