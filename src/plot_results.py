import os
import sys

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec


# Configure Matplotlib styling for high-quality academic publications
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "lines.linewidth": 2.0,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

AERO_COLORS = {
    "primary": "#005F73",
    "secondary": "#0A9396",
    "accent1": "#EE9B00",
    "accent2": "#CA6702",
    "accent3": "#BB3E03",
    "danger": "#AE2012",
    "dark": "#001219",
    "palette": ["#005F73", "#EE9B00", "#94D2BD", "#BB3E03", "#E9D8A6", "#0A9396"]
}


def plot_nominal_learning_curve_episodes(
    csv_path: str = "results/nominal_learning_curve.csv",
    output_dir: str = "results",
):
    """
    Plot nominal learning curve with Episodes on the x-axis.
    """
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    # Estimate episode numbers if not explicitly present (avg episode length ~300 steps)
    if "episodes_approx" in df and df["episodes_approx"].max() > 0:
        x_vals = df["episodes_approx"]
        x_label = "Training Episodes"
    else:
        # Approximate cumulative episodes from timestep and mean episode length
        mean_lens = df["mean_episode_length"].replace(0, 200).fillna(200)
        # Approximate incremental episodes per 10k step block
        approx_eps = np.cumsum(10000.0 / mean_lens)
        x_vals = approx_eps
        x_label = "Training Episodes (Estimated Cumulative)"

    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, height_ratios=[1.2, 1.0], hspace=0.3, wspace=0.25)

    # 1. Episode vs Mean Reward
    ax1 = fig.add_subplot(gs[0, :])
    mean_reward = df["mean_reward"]
    std_reward = df["std_reward"]

    ax1.plot(x_vals, mean_reward, color=AERO_COLORS["primary"], label="PPO Mean Return", lw=2.5)
    ax1.fill_between(
        x_vals,
        mean_reward - std_reward,
        mean_reward + std_reward,
        color=AERO_COLORS["secondary"],
        alpha=0.25,
        label=r"$\pm 1\sigma$ Confidence Interval",
    )
    ax1.axhline(y=200, color=AERO_COLORS["danger"], linestyle=":", lw=1.8, label=r"Solved Threshold (Score $\geq 200$)")
    ax1.axhline(y=0, color="gray", linestyle="-", alpha=0.5, lw=1.0)
    
    ax1.set_title("Nominal PPO Learning Curve: Episode Count vs. Return", fontweight="bold")
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Mean Evaluation Return")
    ax1.grid(True)
    ax1.legend(loc="lower right", framealpha=0.9)

    # 2. Episode vs Success Rate (%)
    ax2 = fig.add_subplot(gs[1, 0])
    success_pct = df["success_rate"] * 100
    ax2.plot(x_vals, success_pct, color=AERO_COLORS["accent2"], lw=2.2, marker="o", markersize=4)
    ax2.set_title("Landing Success Rate (%)", fontweight="bold")
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("Success Rate (%)")
    ax2.set_ylim(-5, 105)
    ax2.grid(True)

    # 3. Episode vs Mean Fuel Conserved
    ax3 = fig.add_subplot(gs[1, 1])
    fuel_rem = df["mean_fuel_remaining"]
    ax3.plot(x_vals, fuel_rem, color=AERO_COLORS["accent3"], lw=2.2, marker="s", markersize=4)
    ax3.set_title("Propellant Conserved at Touchdown", fontweight="bold")
    ax3.set_xlabel(x_label)
    ax3.set_ylabel("Remaining Fuel (kg)")
    ax3.set_ylim(0, 105)
    ax3.grid(True)

    out_path = os.path.join(output_dir, "nominal_learning_curve_episodes.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved nominal learning curve (Episodes) to: {out_path}")


def plot_convergence_bar_chart(summary_csv: str = "results/convergence_summary_table.csv", output_dir: str = "results"):
    """
    Plot Time-to-Convergence bar chart comparing sample efficiency (Steps & Episodes) across hyperparameters.
    """
    if not os.path.exists(summary_csv):
        print(f"Summary CSV not found: {summary_csv}")
        return

    df = pd.read_csv(summary_csv)
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    categories = df["category"].unique()
    labels = [f"{row['parameter']}={row['value']}" for _, row in df.iterrows()]
    steps = df["steps_to_converge"] / 1000.0  # in thousands

    colors = []
    for cat in df["category"]:
        if "Learning" in cat:
            colors.append(AERO_COLORS["primary"])
        elif "Discount" in cat:
            colors.append(AERO_COLORS["accent1"])
        else:
            colors.append(AERO_COLORS["secondary"])

    # Panel 1: Timesteps to Convergence
    bars1 = axes[0].barh(labels, steps, color=colors, edgecolor="black", height=0.6)
    axes[0].set_xlabel("Environment Steps to Converge (Thousands)")
    axes[0].set_title("Sample Efficiency: Timesteps to Convergence", fontweight="bold")
    axes[0].grid(True, axis="x")

    for bar in bars1:
        w = bar.get_width()
        axes[0].text(w + 2, bar.get_y() + bar.get_height()/2, f"{w:.0f}k steps", va="center", fontsize=9)

    # Panel 2: Final Return Achieved
    rewards = df["final_mean_reward"]
    bars2 = axes[1].barh(labels, rewards, color=colors, edgecolor="black", height=0.6)
    axes[1].axvline(x=200, color=AERO_COLORS["danger"], linestyle=":", lw=1.8, label=r"Solved ($Score \geq 200$)")
    axes[1].set_xlabel("Mean Return at Stopping")
    axes[1].set_title("Convergence Quality: Final Return", fontweight="bold")
    axes[1].grid(True, axis="x")
    axes[1].legend(loc="lower right")

    for bar in bars2:
        w = bar.get_width()
        axes[1].text(w + 3, bar.get_y() + bar.get_height()/2, f"{w:.1f}", va="center", fontsize=9)

    plt.suptitle(r"Hyperparameter Convergence Comparison: Time-to-Threshold (Return $\geq 190$)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "convergence_comparison_barplot.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved convergence comparison bar plot to: {out_path}")


def main():
    plot_nominal_learning_curve_episodes()
    if os.path.exists("results/convergence_summary_table.csv"):
        plot_convergence_bar_chart()


if __name__ == "__main__":
    main()
