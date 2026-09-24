# Memory as Maintained Coupling: Kuramoto Model of Synchronization Dynamics

Simulation code for the manuscript:

> **Alvarado, Y. J., Quintero, M., Cardozo-Urdaneta, A., Lossada, C., Marrero-Ponce, Y., Martinez-Ríos, F., Pérez-Castillo, Y., Delgado, A., & González-Paz, L.** (2026). *Memory as Maintained Coupling: A Kuramoto Model Formalization of Synchronization Dynamics in Neural Systems.*

This repository contains the Python code that generates **Figure 2**, **Figure 3**, and **Supplementary Figure S1** of the manuscript.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<TU-USUARIO>/maintained-coupling-kuramoto/blob/main/Figure2_Kuramoto_MainDynamics.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<TU-USUARIO>/maintained-coupling-kuramoto/blob/main/Figure3_Attractor_Landscape.ipynb)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22928314.svg)](https://doi.org/10.5281/zenodo.22928314)

---

## Repository contents

| File | Description |
|------|-------------|
| `Figure2_Kuramoto_MainDynamics.ipynb` | Google Colab notebook — Figure 2 (panels A–H) and Figure S1 |
| `Figure2_Kuramoto_MainDynamics.py`    | Same analysis as a standalone Python script |
| `Figure3_Attractor_Landscape.ipynb`   | Google Colab notebook — Figure 3 (panels A–C) |
| `Figure3_Attractor_Landscape.py`      | Same analysis as a standalone Python script |
| `requirements.txt`                    | Python dependencies |
| `output/`                             | Generated figures (PNG) |

---

## Quick start (Google Colab)

Click any of the Colab badges above to open the notebook directly. Then:

1. In Colab, select **Runtime → Run all**.
2. The simulation will run in approximately 10–15 minutes (Figure 2) and 3–5 minutes (Figure 3).
3. The output figures will appear at the end of the notebook and will also be saved to the Colab working directory.

No installation is required — Colab already includes NumPy, SciPy, Matplotlib, and tqdm.

---

## Local installation

If you prefer to run the scripts locally:

```bash
# Clone the repository
git clone https://github.com/<TU-USUARIO>/maintained-coupling-kuramoto.git
cd maintained-coupling-kuramoto

# Install dependencies
pip install -r requirements.txt

# Run the scripts
python Figure2_Kuramoto_MainDynamics.py
python Figure3_Attractor_Landscape.py
