# Continuous PPO Controller for Mass-Varying Lunar Lander

**TU Delft | Faculty of Aerospace Engineering**  
**Course:** AE4350 Bio-inspired Intelligence  
**Author:** Alvaro Craus Vidal  

---

## Overview

This repository contains the complete implementation, training pipeline, sensitivity analysis, and LaTeX technical report for an autonomous flight controller applied to a **mass-varying Lunar Lander** with dynamic propellant consumption. 

Using **Proximal Policy Optimization (PPO)** with a continuous **Actor-Critic (Adaptive Critic Design)** architecture, the agent learns to continuously modulate the main engine throttle and lateral attitude thrusters to achieve precision soft touchdowns on the lunar surface while adapting to continuous mass loss and conserving propellant.

---

## Key Physical & Algorithmic Features

- **Mass-Varying Multi-Body Dynamics:** Built on an augmented Box2D environment (`CustomLunarLanderContinuous`). As fuel is burned, vehicle density $\rho(t)$ drops linearly from wet density ($\rho_0 = 5.0\,\text{kg/m}^2$, total mass $\approx 4.82\,\text{kg}$) to dry density ($\rho_{\text{dry}} = 2.5\,\text{kg/m}^2$, dry mass $\approx 2.41\,\text{kg}$), doubling control acceleration near touchdown.
- **Continuous 9D State Space:** Horizontal and vertical positions $(x, y)$, velocities $(v_x, v_y)$, pitch angle $\theta$, angular velocity $\omega$, ground contact sensors for both landing legs, and normalized remaining fuel fraction $f(t) = F(t)/F_0$.
- **Continuous 2D Control Space:** 
  - Main engine throttle $u_{\text{main}} \in [0.5, 1.0]$ when active ($u_1 > 0$).
  - Lateral attitude thruster $u_{\text{side}} \in [-1.0, -0.5] \cup [0.5, 1.0]$ with deadband ($|u_2| > 0.5$).
- **Continuous Exploration via Entropy Regularization:** Addresses the chattering and instability of discrete $\epsilon$-greedy heuristics in continuous action spaces by parameterizing a Gaussian policy $\pi(u \mid s) \sim \mathcal{N}(\mu(s), \sigma^2)$ regularized with an entropy bonus $c_{\text{ent}} \mathcal{H}(\pi)$.
- **Potential-Based Reward Shaping & Fuel Penalization:** Guides translational and rotational convergence to the pad while penalizing unnecessary fuel expenditure.

---

## Repository Structure

```text
.
├── README.md                           # Project documentation and quickstart guide
├── requirements.txt                    # Python dependencies
│
├── src/                                # Source code directory
│   ├── custom_lander.py                # Gymnasium environment with mass/fuel dynamics
│   ├── callbacks.py                    # Evaluation and convergence-monitoring callbacks
│   ├── train.py                        # PPO training pipeline with stopping criteria
│   ├── evaluate.py                     # Deterministic policy evaluation & rollout script
│   ├── sensitivity_analysis.py         # Parametric study across alpha, gamma, and c_ent
│   ├── plot_results.py                 # High-resolution LaTeX plotting suite
│   └── main.py                         # Interactive demonstration with visual rendering
│
├── results/                            # CSV logs and generated figures
│   ├── nominal_learning_curve.csv      # Nominal training progression data
│   ├── convergence_sensitivity_analysis.csv # Timestep logs for sensitivity study
│   ├── convergence_summary_table.csv   # Aggregated convergence benchmarks
│   ├── nominal_learning_curve.pdf      # Nominal learning curve (PDF)
│   ├── sensitivity_*.pdf               # Individual sensitivity comparison plots
│   ├── convergence_comparison_barplot.pdf # Summary convergence bar chart
│   └── trajectory_analysis.pdf         # Multi-channel flight trajectory history
│
├── report/                             # Academic technical report (LaTeX)
│   ├── report.tex                      # Master LaTeX document
│   ├── report.pdf                      # Compiled final report
│   ├── report.bib                      # Bibliography references
│   ├── frontmatter/                    # Title page, TOC, nomenclature
│   ├── mainmatter/                     # Main report chapters (Sections 1–7)
│   ├── appendix/                       # Flowcharts and parameter tables
│   └── figures/                        # High-resolution vector figures
│
├── models/                             # Saved neural network checkpoints
└── logs/                               # TensorBoard training monitor logs
```

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/acrausvidal/BioInspired_2026_lander_project.git
cd BioInspired_2026_lander_project
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# On Linux / macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage Guide

### 1. Run Interactive Visual Demonstration
To observe the trained PPO agent landing the spacecraft with real-time Pygame visualization:
```bash
python src/main.py
```

### 2. Train the Nominal PPO Agent
Train the baseline PPO controller on 4 parallelized environments:
```bash
python src/train.py --max-timesteps 300000 --learning-rate 0.0003 --gamma 0.99 --ent-coef 0.01
```

### 3. Evaluate Policy Performance
Run a Monte Carlo evaluation campaign over $N$ deterministic flight trials:
```bash
# Evaluate 50 test episodes and display quantitative metrics
python src/evaluate.py --episodes 50

# Evaluate with visual rendering enabled
python src/evaluate.py --episodes 5 --render
```

### 4. Run Hyperparameter Sensitivity Analysis
Execute the automated parametric sweep across learning rates ($\alpha$), discount factors ($\gamma$), and entropy coefficients ($c_{\text{ent}}$):
```bash
python src/sensitivity_analysis.py
```

### 5. Re-generate All Figures
Regenerate all publication-quality LaTeX figures and sync them to `results/` and `report/figures/`:
```bash
python src/plot_results.py
```

### 6. Compile the Technical Report
Compile the LaTeX report using `latexmk` or `pdflatex` + `biber`:
```bash
cd report
latexmk -pdf report.tex
```

---

## Summary of Results

| Experiment / Metric | Convergence Steps | Mean Return $\mathbb{E}[R]$ | Safe Landing Rate | Remaining Fuel |
| :--- | :---: | :---: | :---: | :---: |
| **Nominal Baseline** ($\alpha=3\cdot 10^{-4}, \gamma=0.99, c_{\text{ent}}=0.01$) | **90,000** | $+196.89 \pm 38.2$ | **93.3%** | **49.1%** |
| **Learning Rate** $\alpha = 10^{-4}$ | 190,000 | $+210.61 \pm 31.4$ | 100.0% | 42.7% |
| **Learning Rate** $\alpha = 10^{-3}$ | 70,000 | $+194.87 \pm 42.1$ | 100.0% | 40.8% |
| **Discount Factor** $\gamma = 0.95$ (Myopic) | *Did not converge* | $-115.76 \pm 88.4$ | 0.0% | 0.0% |
| **Discount Factor** $\gamma = 0.999$ (Far-Sighted) | 150,000 | $+198.94 \pm 36.5$ | 93.3% | **64.9%** |
| **Entropy Coeff.** $c_{\text{ent}} = 0.00$ (No exploration) | 120,000 | $+190.36 \pm 40.8$ | 100.0% | 31.3% |
| **Entropy Coeff.** $c_{\text{ent}} = 0.05$ (High exploration) | 90,000 | $+196.65 \pm 34.7$ | 100.0% | 42.8% |

---

## References

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
2. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
3. van Kampen, E. (2025). *AE4350 Bio-inspired Intelligence Lecture Notes*. Delft University of Technology.
4. Towers, M., et al. (2023). *Gymnasium: A Standard Interface for Reinforcement Learning Environments*. arXiv:2407.17032.