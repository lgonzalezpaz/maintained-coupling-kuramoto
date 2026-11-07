# -*- coding: utf-8 -*-
"""
make_notebooks.py
=================
Generates standard valid Jupyter Notebooks (nbformat v4) for:
  - Figure2_Kuramoto_MainDynamics.ipynb
  - Figure3_Attractor_Landscape.ipynb
"""

import json
import pathlib

ROOT_DIR = pathlib.Path(__file__).resolve().parent


def create_notebook(py_filename, ipynb_filename, title, description):
    py_path = ROOT_DIR / py_filename
    ipynb_path = ROOT_DIR / ipynb_filename

    with open(py_path, 'r', encoding='utf-8') as f:
        code_content = f.read()

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# {title}\n",
                "\n",
                f"{description}\n",
                "\n",
                "### Manuscript:\n",
                "> **Alvarado, Y. J., Quintero, M., Cardozo-Urdaneta, A., Lossada, C., Marrero-Ponce, Y., Martinez-Ríos, F., Pérez-Castillo, Y., Delgado, A., & González-Paz, L.** (2026). *Memory as Maintained Coupling: A Kuramoto Model Formalization of Synchronization Dynamics in Neural Systems.*\n",
                "\n",
                "### Instructions:\n",
                "Run all cells to execute the simulations, plot publication-grade figures (PNG and vector PDF), and export numerical metrics to `results/`.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in code_content.splitlines()]
        }
    ]

    nb_data = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open(ipynb_path, 'w', encoding='utf-8') as f:
        json.dump(nb_data, f, indent=1)

    print(f"Created {ipynb_filename} successfully.")


if __name__ == "__main__":
    create_notebook(
        "Figure2_Kuramoto_MainDynamics.py",
        "Figure2_Kuramoto_MainDynamics.ipynb",
        "Figure 2 & Figure S1 — Kuramoto Model Dynamics and Developmental Pruning",
        "Reproduces Figure 2 (Panels A–H) and Supplementary Figure S1, implementing endogenous Hebbian acquisition, retention, cue-triggered recall, disruptive readout, progressive CA3 pruning, clustering coefficients, and stochastic resonance Information Transfer Efficiency (ITE)."
    )
    create_notebook(
        "Figure3_Attractor_Landscape.py",
        "Figure3_Attractor_Landscape.ipynb",
        "Figure 3 & Figure S2 — Attractor Energy Landscape and Kramers Survival Dynamics",
        "Reproduces Figure 3 (Panels A–C) and Supplementary Figure S2, implementing true potential energy landscapes E(Delta-phi) from Equation (14), Kaplan-Meier RMST survival analysis, 2x2 factorial scaling design, and Kramers escape rate scaling verification."
    )
