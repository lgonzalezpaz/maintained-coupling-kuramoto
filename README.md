# Memory as Maintained Coupling: A Kuramoto Model Formalization of Synchronization Dynamics and Developmental Pruning in Neural Systems

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22928314.svg)](https://doi.org/10.5281/zenodo.22928314)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Official open-source simulation library, reproducibility suite, and manuscript source repository for:

> **Alvarado, Y. J., Quintero, M., Cardozo-Urdaneta, A., Lossada, C., Marrero-Ponce, Y., Martinez-Ríos, F., Pérez-Castillo, Y., Delgado, A., & González-Paz, L. (2026).** *Memory as maintained coupling: a Kuramoto model formalization of synchronization dynamics and developmental pruning in neural systems.*

---

## Overview & Key Scientific Contributions (v2.0.0)

This repository implements a connectome-calibrated, sparse, distance-dependent Kuramoto phase-oscillator framework with endogenous Hebbian plasticity, structural synaptic remodeling, progressive developmental pruning, and stochastic Langevin dynamics calibrated to postnatal mouse hippocampal CA3 connectomic measurements ([Vargas-Barroso et al., *Nat. Commun.* 2026](https://doi.org/10.1038/s41467-026-71914-x)).

### Core Mathematical & Algorithmic Features
1. **Endogenous Hebbian Consolidation vs. Plasticity-Ablated Control (`Figure 2A–B`):**
   - Strictly separates transient external task induction drive $I_{\text{ext}}(t)$ ($t \in [30, 70]$ a.u.) from endogenous inter-population synaptic weight co-evolution $K_{\text{cross},ij}(t)$.
   - Demonstrates autonomous maintenance of phase-locked synchrony after stimulus offset ($r_{\text{global}} = 0.988$ vs. $0.556$ in `Plasticity OFF` ablation) and rapid pattern reinstatement under a partial recall cue at $t \in [135, 145]$ a.u. ($r_{\text{global}} = 0.963$ vs. $0.498$).
2. **Disruptive Readout Assay & Von Neumann Non-Separation Criterion (`Figure 2F`):**
   - Operationalizes von Neumann's and Mangalam's physical storage criterion by demonstrating that probing a consolidated oscillatory trace requires active phase execution that inevitably induces transient synaptic depression ($\Delta K = -0.083\text{ a.u.}, P = 1.4 \times 10^{-22}$).
3. **Strict Monotonic Developmental Pruning & Motif Null Models (`Figure 2C–D`):**
   - Enforces a nested edge-elimination hierarchy ($E(t_2) \subset E(t_1)$) with exact symmetrized off-diagonal density matching postnatal CA3 development: **6.37%** (P7–8) $\to$ **2.79%** (P18–25) $\to$ **2.17%** (P45–50) ($2.94\times$ sparsification, $R^2 = 1.000$).
   - Benchmarks Watts–Strogatz binary ($C = 0.0752$) and Onnela weighted ($C^W = 0.0267$) clustering against Erdős–Rényi 95% confidence bands, and verifies recurrent disynaptic motif enrichment against $n_{\text{null}} = 300$ degree-preserving Maslov–Sneppen surrogates ($Z = +10.85, P = 0.0033$, explicitly distinguished from the empirical CA3 $P = 0.03$).
4. **First-Passage Consolidated Attractor Escape & Operational Stochastic Resonance (`Figure 2G–H`):**
   - Initializes Langevin trajectories directly at the consolidated attractor $\theta^*$ ($n = 30$ trials per $\sigma$), identifying a critical escape threshold at $\sigma_c = 0.70$.
   - Computes task-based subthreshold cued **Information Transfer Efficiency (ITE)** directly from simulated trajectories, revealing an inverted-U stochastic-resonance peak at intermediate noise ($\sigma = 0.40, \text{ITE} = 0.145$ vs. $0.000$ at $\sigma = 0$).
5. **Exact Potential Energy Landscape, $2 \times 2$ Factorial Survival Analysis, and Kramers Scaling (`Figure 3` & `Figure S2`):**
   - Evaluates the exact Kuramoto Lyapunov potential $E(\theta) = -\frac{1}{2N}\sum_{i,j} A_{ij}\cos(\theta_j - \theta_i)$ across developmental topologies ($\Delta E = 6.25 \to 14.28\text{ a.u.}$ in Model-Pruned; $5.31 \to 11.42\text{ a.u.}$ in Empirical-Calibrated).
   - Quantifies right-censored escape times via Kaplan–Meier **Restricted Mean Survival Time (RMST)** across a $2 \times 2$ factorial design ($\rho \times K_0$) plus a **Matched Total Synaptic Budget Control** ($K_0 \cdot \rho = 43.4$, yielding a $2.5\times$ stability gain) and verifies Arrhenius–Kramers barrier scaling ($\ln \tau_{\text{RMST}} = -3.56 + 12.15/\sigma^2, R^2 = 0.8507, P = 0.0257$).
6. **Multi-Seed Sensitivity Grid, Global PRCC, and Interleaved Replay (`Figure S1`):**
   - Evaluates a $7 \times 7$ $(\eta, \lambda_K)$ grid across $10$ matched random seeds per cell ($490$ runs), global Partial Rank Correlation Coefficients (PRCC) across $200$ Latin-hypercube samples, and sequential 4-pattern learning stabilized by sleep-like interleaved replay ($34.6\% \to 11.8\%$ overlap).

---

## Repository Structure

```text
maintained-coupling-kuramoto/
├── simulation_core.py                  # Unified mathematical & numerical simulation engine
├── Figure1_Conceptual_Framework.py     # Generates Figure 1 (PNG 300 DPI & vector PDF)
├── Figure2_Kuramoto_MainDynamics.py    # Generates Figure 2 & Supplementary Figure S1 + CSV/JSON
├── Figure3_Attractor_Landscape.py      # Generates Figure 3 & Supplementary Figure S2 + CSV/JSON
├── aggregate_metrics.py                # Compiles master results/summary_metrics.json
├── make_notebooks.py                   # Generates synchronized Jupyter/Colab notebooks (.ipynb)
├── build_manuscript_docx.py            # Generates publication-ready Word (.docx) documents
├── Figure2_Kuramoto_MainDynamics.ipynb # Interactive Jupyter/Colab notebook for Figure 2 & S1
├── Figure3_Attractor_Landscape.ipynb   # Interactive Jupyter/Colab notebook for Figure 3 & S2
├── Manuscript.docx                     # Complete revised manuscript with embedded figures & tables
├── Manuscript_Text_Only.docx           # Manuscript text & tables with figure placeholders
├── Supplementary Material.docx         # Complete Supplementary Material (Methods, Figs S1-S2, Tables S1-S4)
├── Response_to_Reviewers.docx          # Point-by-point response letter to Reviewers 1 & 2
├── results/
│   ├── summary_metrics.json            # Master verified numerical metrics & statistical outputs
│   ├── developmental_pruning_trajectory.csv
│   ├── noise_stability_ite.csv
│   ├── sensitivity_grid_multiseed.csv
│   ├── global_prcc_sensitivity.csv
│   ├── two_by_two_factorial_escape.csv
│   └── kramers_scaling_analysis.csv
├── Figure1.png / Figure1.pdf           # Main Figure 1
├── Figure2.png / Figure2.pdf           # Main Figure 2
├── Figure3.png / Figure3.pdf           # Main Figure 3
├── FigureS1.png / FigureS1.pdf         # Supplementary Figure S1
├── FigureS2.png / FigureS2.pdf         # Supplementary Figure S2
├── requirements.txt                    # Python package dependencies
├── environment.yml                     # Conda environment specification
├── CITATION.cff                        # Citation metadata
├── .zenodo.json                        # Zenodo archival metadata
└── LICENSE                             # MIT License
```

---

## Reproducing All Simulations, Figures, and Manuscript Files

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full simulation pipeline and generate all figures & CSV/JSON tables
python Figure1_Conceptual_Framework.py
python Figure2_Kuramoto_MainDynamics.py
python Figure3_Attractor_Landscape.py
python aggregate_metrics.py

# 3. Rebuild Jupyter notebooks and Word (.docx) manuscripts
python make_notebooks.py
python build_manuscript_docx.py
```

All simulations use fixed, documented random seeds (`seed = 42` base offset) and execute deterministically in ~35 seconds on a standard desktop CPU.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
