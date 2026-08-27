import os
import sys
import shutil

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

# Enable full LaTeX text rendering and Computer Modern / Serif typography with larger base font sizes
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{siunitx}",
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11.5,
    "figure.titlesize": 15,
    "figure.dpi": 300,
    "lines.linewidth": 2.4,
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
    "palette": ["#1D3557", "#457B9D", "#2A9D8F", "#E76F51", "#F4A261", "#264653"]
}


def _save_multi_format(fig, base_path_no_ext: str):
    """Save figure in both vector PDF and raster PNG to both results/ and report/figures/."""
    pdf_path = f"{base_path_no_ext}.pdf"
    png_path = f"{base_path_no_ext}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {pdf_path} and {png_path}")

    # Synchronize with report/figures if saving in results
    filename = os.path.basename(base_path_no_ext)
    report_fig_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "report", "figures")
    if os.path.exists(report_fig_dir):
        report_pdf = os.path.join(report_fig_dir, f"{filename}.pdf")
        report_png = os.path.join(report_fig_dir, f"{filename}.png")
        shutil.copyfile(pdf_path, report_pdf)
        shutil.copyfile(png_path, report_png)
        print(f"Synced to report: {report_pdf}")


def format_param_label(category: str, param: str, value: str) -> str:
    """Format hyperparameters into clean LaTeX mathematical expressions."""
    try:
        val_float = float(value)
    except ValueError:
        return f"${value}$"

    if "Learning" in category or "learning_rate" in param:
        if abs(val_float - 0.0001) < 1e-6:
            return r"$\alpha = 10^{-4}$"
        elif abs(val_float - 0.0003) < 1e-6:
            return r"$\alpha = 3 \times 10^{-4}$"
        elif abs(val_float - 0.001) < 1e-6:
            return r"$\alpha = 10^{-3}$"
        return rf"$\alpha = {val_float}$"
    elif "Discount" in category or "gamma" in param:
        return rf"$\gamma = {val_float}$"
    else:
        return rf"$c_{{\text{{ent}}}} = {val_float}$"


def plot_nominal_learning_curve(csv_path: str = "results/nominal_learning_curve.csv", output_dir: str = "results"):
    """
    Plot nominal agent learning curve vs timesteps with full LaTeX formatting and large readable fonts.
    """
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(8.2, 5.4))
    gs = GridSpec(2, 2, height_ratios=[1.2, 1.0], hspace=0.34, wspace=0.28)

    # 1. Main Return vs Timesteps
    ax1 = fig.add_subplot(gs[0, :])
    timesteps = df["timestep"]
    mean_reward = df["mean_reward"]
    std_reward = df["std_reward"]

    ax1.plot(timesteps, mean_reward, color=AERO_COLORS["primary"], label=r"PPO Mean Return $\mathbb{E}[R_t]$", lw=2.6)
    ax1.fill_between(
        timesteps,
        mean_reward - std_reward,
        mean_reward + std_reward,
        color=AERO_COLORS["secondary"],
        alpha=0.25,
        label=r"$\pm 1\sigma$ Confidence Interval",
    )
    ax1.axhline(y=200, color=AERO_COLORS["danger"], linestyle=":", lw=2.0, label=r"Solved Benchmark ($R \geq 200$)")
    ax1.axhline(y=0, color="gray", linestyle="-", alpha=0.5, lw=0.8)
    
    ax1.set_xlabel(r"Environment Timesteps ($t$)", fontsize=13)
    ax1.set_ylabel(r"Mean Return $\mathbb{E}[R]$", fontsize=13)
    ax1.grid(True)
    ax1.legend(loc="lower right", framealpha=0.92, fontsize=11)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x/1000)}k"))

    # 2. Success Rate (%)
    ax2 = fig.add_subplot(gs[1, 0])
    success_pct = df["success_rate"] * 100
    ax2.plot(timesteps, success_pct, color=AERO_COLORS["accent2"], lw=2.4, marker="o", markersize=4)
    ax2.set_xlabel(r"Environment Timesteps ($t$)", fontsize=12)
    ax2.set_ylabel(r"Safe Landing Rate (\%)", fontsize=12)
    ax2.set_ylim(-5, 105)
    ax2.grid(True)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x/1000)}k"))

    # 3. Propellant Remaining
    ax3 = fig.add_subplot(gs[1, 1])
    fuel_rem = df["mean_fuel_remaining"]
    ax3.plot(timesteps, fuel_rem, color=AERO_COLORS["accent3"], lw=2.4, marker="s", markersize=4)
    ax3.set_xlabel(r"Environment Timesteps ($t$)", fontsize=12)
    ax3.set_ylabel(r"Remaining Fuel $F(t)$ (\%)", fontsize=12)
    ax3.set_ylim(0, 105)
    ax3.grid(True)
    ax3.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x/1000)}k"))

    _save_multi_format(fig, os.path.join(output_dir, "nominal_learning_curve"))
    plt.close()


def plot_nominal_learning_curve_episodes(csv_path: str = "results/nominal_learning_curve.csv", output_dir: str = "results"):
    """
    Plot nominal agent learning curve vs training episodes with full LaTeX formatting.
    """
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    if "episodes_approx" in df and df["episodes_approx"].max() > 0:
        x_vals = df["episodes_approx"]
        x_label = r"Training Episodes ($N_{\text{ep}}$)"
    else:
        mean_lens = df["mean_episode_length"].replace(0, 200).fillna(200)
        x_vals = np.cumsum(10000.0 / mean_lens)
        x_label = r"Estimated Training Episodes ($N_{\text{ep}}$)"

    fig = plt.figure(figsize=(8.2, 5.4))
    gs = GridSpec(2, 2, height_ratios=[1.2, 1.0], hspace=0.34, wspace=0.28)

    ax1 = fig.add_subplot(gs[0, :])
    mean_reward = df["mean_reward"]
    std_reward = df["std_reward"]

    ax1.plot(x_vals, mean_reward, color=AERO_COLORS["primary"], label=r"PPO Mean Return $\mathbb{E}[R]$", lw=2.6)
    ax1.fill_between(
        x_vals,
        mean_reward - std_reward,
        mean_reward + std_reward,
        color=AERO_COLORS["secondary"],
        alpha=0.25,
        label=r"$\pm 1\sigma$ Confidence Interval",
    )
    ax1.axhline(y=200, color=AERO_COLORS["danger"], linestyle=":", lw=2.0, label=r"Solved Benchmark ($R \geq 200$)")
    ax1.axhline(y=0, color="gray", linestyle="-", alpha=0.5, lw=0.8)
    
    ax1.set_xlabel(x_label, fontsize=13)
    ax1.set_ylabel(r"Mean Return $\mathbb{E}[R]$", fontsize=13)
    ax1.grid(True)
    ax1.legend(loc="lower right", framealpha=0.92, fontsize=11)

    ax2 = fig.add_subplot(gs[1, 0])
    success_pct = df["success_rate"] * 100
    ax2.plot(x_vals, success_pct, color=AERO_COLORS["accent2"], lw=2.4, marker="o", markersize=4)
    ax2.set_xlabel(x_label, fontsize=12)
    ax2.set_ylabel(r"Safe Landing Rate (\%)", fontsize=12)
    ax2.set_ylim(-5, 105)
    ax2.grid(True)

    ax3 = fig.add_subplot(gs[1, 1])
    fuel_rem = df["mean_fuel_remaining"]
    ax3.plot(x_vals, fuel_rem, color=AERO_COLORS["accent3"], lw=2.4, marker="s", markersize=4)
    ax3.set_xlabel(x_label, fontsize=12)
    ax3.set_ylabel(r"Remaining Fuel (\%)", fontsize=12)
    ax3.set_ylim(0, 105)
    ax3.grid(True)

    _save_multi_format(fig, os.path.join(output_dir, "nominal_learning_curve_episodes"))
    plt.close()


def plot_sensitivity_analysis(csv_path: str = "results/convergence_sensitivity_analysis.csv", output_dir: str = "results"):
    """
    Plot hyperparameter sensitivity comparison from the convergence-criteria data.
    Curves end at each configuration's exact convergence stopping point matching Table tab:sensitivity_summary.
    Plots use compact dimensions and large readable typography for 0.75 linewidth inclusion.
    """
    # Fallback to standard sensitivity csv if convergence csv not found
    if not os.path.exists(csv_path):
        csv_path = "results/sensitivity_analysis.csv"
        if not os.path.exists(csv_path):
            return

    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    categories = ["Learning Rate", "Discount Factor", "Exploration-Exploitation (Entropy)"]
    file_suffixes = {
        "Learning Rate": "learning_rate",
        "Discount Factor": "gamma",
        "Exploration-Exploitation (Entropy)": "ent_coef",
    }

    # High-contrast aerospace palette
    colors_list = ["#005F73", "#EE9B00", "#AE2012", "#0A9396"]

    # 1. Combined 3-Panel Dashboard
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)

    for ax_idx, cat in enumerate(categories):
        ax = axes[ax_idx]
        cat_df = df[df["category"] == cat]
        param_values = cat_df["value"].unique()

        for idx, val in enumerate(param_values):
            sub = cat_df[cat_df["value"] == val].sort_values("timestep")
            color = colors_list[idx % len(colors_list)]
            label_name = format_param_label(cat, sub["parameter"].iloc[0], val)
            ax.plot(sub["timestep"], sub["mean_reward"], color=color, label=label_name, lw=2.5)
            ax.fill_between(
                sub["timestep"],
                sub["mean_reward"] - 0.5 * sub["std_reward"],
                sub["mean_reward"] + 0.5 * sub["std_reward"],
                color=color,
                alpha=0.18,
            )

        ax.axhline(y=200, color="#AE2012", linestyle=":", lw=1.8, label=r"Solved ($200$)")
        ax.axhline(y=0, color="black", linestyle="-", lw=0.8, alpha=0.4)
        ax.set_xlabel(r"Timesteps ($t$)", fontsize=13)
        if ax_idx == 0:
            ax.set_ylabel(r"Mean Return $\mathbb{E}[R]$", fontsize=14)
        ax.grid(True)
        ax.legend(loc="lower right", framealpha=0.92, fontsize=11)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x/1000)}k"))

    plt.tight_layout()
    _save_multi_format(fig, os.path.join(output_dir, "sensitivity_comparison_dashboard"))
    plt.close()

    # 2. Individual Dedicated Plots for Each Parameter (compact size, large readable font at 0.75 linewidth)
    for cat in categories:
        cat_df = df[df["category"] == cat]
        param_name = cat_df["parameter"].iloc[0]
        file_tag = file_suffixes.get(cat, param_name)

        fig_single, ax_single = plt.subplots(figsize=(6.2, 4.0))
        param_values = cat_df["value"].unique()

        for idx, val in enumerate(param_values):
            sub = cat_df[cat_df["value"] == val].sort_values("timestep")
            color = colors_list[idx % len(colors_list)]
            label_name = format_param_label(cat, param_name, val)
            ax_single.plot(sub["timestep"], sub["mean_reward"], color=color, label=label_name, lw=2.6)
            ax_single.fill_between(
                sub["timestep"],
                sub["mean_reward"] - 0.5 * sub["std_reward"],
                sub["mean_reward"] + 0.5 * sub["std_reward"],
                color=color,
                alpha=0.18,
            )

        ax_single.axhline(y=200, color="#AE2012", linestyle=":", lw=2.0, label=r"Solved ($R \geq 200$)")
        ax_single.axhline(y=0, color="gray", linestyle="-", alpha=0.6, lw=0.9)
        
        ax_single.set_xlabel(r"Environment Timesteps ($t$)", fontsize=14)
        ax_single.set_ylabel(r"Mean Evaluation Return $\mathbb{E}[R]$", fontsize=14)
        ax_single.tick_params(axis="both", labelsize=12.5)
        ax_single.grid(True)
        ax_single.legend(loc="lower right", framealpha=0.94, fontsize=12)
        ax_single.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x/1000)}k"))

        plt.tight_layout()
        _save_multi_format(fig_single, os.path.join(output_dir, f"sensitivity_{file_tag}"))
        plt.close()


def plot_trajectory_profile(model_path: str = "models/best_model.zip", output_dir: str = "results"):
    """
    Run deterministic rollout and plot physical states and controls with full LaTeX typography.
    """
    from stable_baselines3 import PPO
    from src.custom_lander import CustomLunarLanderContinuous

    if not os.path.exists(model_path):
        if os.path.exists(model_path.replace(".zip", "")):
            model_path = model_path.replace(".zip", "")
        else:
            return

    env = CustomLunarLanderContinuous()
    model = PPO.load(model_path)

    obs, info = env.reset(seed=42)
    done = False

    time_steps = []
    positions_x = []
    positions_y = []
    velocities_x = []
    velocities_y = []
    angles = []
    main_throttles = []
    side_throttles = []
    fuels = []
    masses = []

    step = 0
    dt = 1.0 / 50.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        t = step * dt
        time_steps.append(t)
        positions_x.append(obs[0])
        positions_y.append(obs[1])
        velocities_x.append(obs[2])
        velocities_y.append(obs[3])
        angles.append(np.degrees(obs[4]))
        main_throttles.append(max(0.0, action[0]))
        side_throttles.append(action[1])
        fuels.append(info.get("fuel_remaining", 0.0))
        masses.append(info.get("lander_mass", 0.0))

        step += 1
        done = terminated or truncated

    env.close()

    time_steps = np.array(time_steps)
    fig, axes = plt.subplots(3, 2, figsize=(9.5, 7.2), sharex=True)

    # Subplot 1: Altitude & Horizontal
    axes[0, 0].plot(time_steps, positions_y, color=AERO_COLORS["primary"], lw=2.4, label=r"Altitude $y(t)$")
    axes[0, 0].plot(time_steps, positions_x, color=AERO_COLORS["accent1"], lw=2.2, linestyle="--", label=r"Lateral $x(t)$")
    axes[0, 0].set_ylabel(r"Position (Normalized)", fontsize=12)
    axes[0, 0].tick_params(axis="both", labelsize=11)
    axes[0, 0].grid(True)
    axes[0, 0].legend(fontsize=11)

    # Subplot 2: Velocities
    axes[0, 1].plot(time_steps, velocities_y, color=AERO_COLORS["danger"], lw=2.4, label=r"Vertical $v_y(t)$")
    axes[0, 1].plot(time_steps, velocities_x, color=AERO_COLORS["secondary"], lw=2.2, linestyle="--", label=r"Lateral $v_x(t)$")
    axes[0, 1].set_ylabel(r"Velocity (Normalized)", fontsize=12)
    axes[0, 1].tick_params(axis="both", labelsize=11)
    axes[0, 1].grid(True)
    axes[0, 1].legend(fontsize=11)

    # Subplot 3: Orientation Angle
    axes[1, 0].plot(time_steps, angles, color=AERO_COLORS["accent2"], lw=2.4)
    axes[1, 0].axhline(y=0, color="gray", linestyle=":", alpha=0.7)
    axes[1, 0].set_ylabel(r"Pitch Angle $\theta$ (\si{\degree})", fontsize=12)
    axes[1, 0].tick_params(axis="both", labelsize=11)
    axes[1, 0].grid(True)

    # Subplot 4: Continuous Throttles
    axes[1, 1].plot(time_steps, main_throttles, color=AERO_COLORS["primary"], lw=2.2, label=r"Main $u_{\text{main}}(t)$")
    axes[1, 1].plot(time_steps, side_throttles, color=AERO_COLORS["accent3"], lw=2.0, linestyle="--", label=r"Lateral $u_{\text{side}}(t)$")
    axes[1, 1].set_ylabel(r"Control Action $u(t)$", fontsize=12)
    axes[1, 1].tick_params(axis="both", labelsize=11)
    axes[1, 1].grid(True)
    axes[1, 1].legend(fontsize=11)

    # Subplot 5: Propellant Depletion
    axes[2, 0].plot(time_steps, fuels, color=AERO_COLORS["accent3"], lw=2.6)
    axes[2, 0].set_xlabel(r"Mission Flight Time $t$ (\si{\second})", fontsize=12.5)
    axes[2, 0].set_ylabel(r"Fuel Remaining $F(t)$ (\%)", fontsize=12)
    axes[2, 0].tick_params(axis="both", labelsize=11)
    axes[2, 0].grid(True)

    # Subplot 6: Mass-Varying System
    axes[2, 1].plot(time_steps, masses, color=AERO_COLORS["secondary"], lw=2.6)
    axes[2, 1].set_xlabel(r"Mission Flight Time $t$ (\si{\second})", fontsize=12.5)
    axes[2, 1].set_ylabel(r"Total Mass $m(t)$ (\si{\kilogram})", fontsize=12)
    axes[2, 1].tick_params(axis="both", labelsize=11)
    axes[2, 1].grid(True)

    plt.tight_layout()
    _save_multi_format(fig, os.path.join(output_dir, "trajectory_analysis"))
    plt.close()


def plot_convergence_bar_chart(summary_csv: str = "results/convergence_summary_table.csv", output_dir: str = "results"):
    """
    Publication-quality horizontal bar chart comparing sample efficiency (steps to converge)
    and convergence return quality across hyperparameter settings.
    Removes the y-labels of the right plot, brings subplots closer with shared y-axis, and uses large fonts.
    """
    if not os.path.exists(summary_csv):
        return

    raw_df = pd.read_csv(summary_csv)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Filter out runs that failed to converge
    df = raw_df[raw_df["converged"] == True].copy()

    category_order = ["Learning Rate", "Discount Factor", "Exploration-Exploitation (Entropy)"]
    df["cat_rank"] = df["category"].map(lambda c: category_order.index(c) if c in category_order else 99)
    df = df.sort_values(by=["cat_rank", "steps_to_converge"], ascending=[True, False]).reset_index(drop=True)

    labels = [format_param_label(row["category"], row["parameter"], row["value"]) for _, row in df.iterrows()]
    steps_k = df["steps_to_converge"] / 1000.0
    returns = df["final_mean_reward"]

    category_colors = {
        "Learning Rate": "#1D3557",
        "Discount Factor": "#457B9D",
        "Exploration-Exploitation (Entropy)": "#2A9D8F",
    }
    colors = [category_colors.get(cat, "#457B9D") for cat in df["category"]]

    # Compact layout with close subplots (small wspace)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.2), gridspec_kw={"wspace": 0.08})
    y_pos = np.arange(len(df))

    # Panel 1: Steps to Convergence (Sample Efficiency)
    bars1 = axes[0].barh(y_pos, steps_k, color=colors, edgecolor="none", height=0.62, alpha=0.92)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(labels, fontsize=13)
    axes[0].set_xlabel(r"Steps to Converge ($10^3$)", fontsize=13.5)
    axes[0].set_xlim(0, 240)
    axes[0].tick_params(axis="x", labelsize=12)
    axes[0].grid(True, axis="x", linestyle="--", alpha=0.45)

    for bar in bars1:
        w = bar.get_width()
        axes[0].text(
            w + 4.0,
            bar.get_y() + bar.get_height() / 2,
            rf"${w:.0f}\,\text{{k}}$",
            va="center",
            ha="left",
            fontsize=12,
            fontweight="bold",
        )

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=category_colors["Learning Rate"], edgecolor="none", label=r"$\alpha$"),
        plt.Rectangle((0, 0), 1, 1, facecolor=category_colors["Discount Factor"], edgecolor="none", label=r"$\gamma$"),
        plt.Rectangle((0, 0), 1, 1, facecolor=category_colors["Exploration-Exploitation (Entropy)"], edgecolor="none", label=r"$c_{\text{ent}}$"),
    ]
    axes[0].legend(handles=legend_elements, loc="upper right", framealpha=0.94, fontsize=11)

    # Panel 2: Final Return Achieved at Stopping (No y-labels, closer layout)
    bars2 = axes[1].barh(y_pos, returns, color=colors, edgecolor="none", height=0.62, alpha=0.92)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels([])  # Remove y-labels to avoid duplication
    axes[1].tick_params(left=False, labelleft=False)
    axes[1].set_xlabel(r"Mean Return at Convergence $\mathbb{E}[R]$", fontsize=13.5)
    axes[1].set_xlim(175, 224)
    axes[1].tick_params(axis="x", labelsize=12)
    axes[1].grid(True, axis="x", linestyle="--", alpha=0.45)

    axes[1].axvline(x=200, color="#AE2012", linestyle="--", lw=1.8, label=r"Solved ($R \geq 200$)")
    axes[1].axvline(x=190, color="#E76F51", linestyle=":", lw=1.8, label=r"Target ($R \geq 190$)")
    axes[1].legend(loc="lower left", framealpha=0.94, fontsize=11)

    for bar in bars2:
        w = bar.get_width()
        axes[1].text(
            w + 0.9,
            bar.get_y() + bar.get_height() / 2,
            rf"${w:.1f}$",
            va="center",
            ha="left",
            fontsize=12,
            fontweight="bold",
        )

    fig.text(
        0.5,
        -0.04,
        r"\textit{Note: The short planning horizon configuration ($\gamma = 0.95$) failed to converge within the 200k safety limit and is omitted.}",
        ha="center",
        fontsize=10,
        color="#333333",
    )

    _save_multi_format(fig, os.path.join(output_dir, "convergence_comparison_barplot"))
    plt.close()


def main():
    plot_nominal_learning_curve()
    plot_nominal_learning_curve_episodes()
    plot_sensitivity_analysis()
    plot_trajectory_profile()
    if os.path.exists("results/convergence_summary_table.csv"):
        plot_convergence_bar_chart()


if __name__ == "__main__":
    main()
