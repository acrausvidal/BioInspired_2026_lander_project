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


def plot_nominal_learning_curve(csv_path: str = "results/nominal_learning_curve.csv", output_dir: str = "results"):
    """
    Plot the nominal agent learning curve proving the learning effect and convergence.
    """
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}. Run src/train.py first.")
        return

    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, height_ratios=[1.2, 1.0], hspace=0.3, wspace=0.25)

    # 1. Main Learning Curve: Timestep vs Mean Reward
    ax1 = fig.add_subplot(gs[0, :])
    timesteps = df["timestep"]
    mean_reward = df["mean_reward"]
    std_reward = df["std_reward"]

    ax1.plot(timesteps, mean_reward, color=AERO_COLORS["primary"], label="PPO Mean Reward", lw=2.5)
    ax1.fill_between(
        timesteps,
        mean_reward - std_reward,
        mean_reward + std_reward,
        color=AERO_COLORS["secondary"],
        alpha=0.25,
        label=r"$\pm 1\sigma$ Confidence Interval",
    )
    ax1.axhline(y=200, color=AERO_COLORS["danger"], linestyle=":", lw=1.8, label=r"Solved Threshold (Score $\geq 200$)")
    ax1.axhline(y=0, color="gray", linestyle="-", alpha=0.5, lw=1.0)
    
    ax1.set_title("Nominal Training Convergence: Proximal Policy Optimization on Mass-Varying Lunar Lander", fontweight="bold")
    ax1.set_xlabel("Environment Timesteps")
    ax1.set_ylabel("Mean Evaluation Return")
    ax1.grid(True)
    ax1.legend(loc="lower right", framealpha=0.9)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x/1000)}k"))

    # 2. Success Rate (%) over training
    ax2 = fig.add_subplot(gs[1, 0])
    success_pct = df["success_rate"] * 100
    ax2.plot(timesteps, success_pct, color=AERO_COLORS["accent2"], lw=2.2, marker="o", markersize=4)
    ax2.set_title("Safe Landing Success Rate (%)", fontweight="bold")
    ax2.set_xlabel("Environment Timesteps")
    ax2.set_ylabel("Success Rate (%)")
    ax2.set_ylim(-5, 105)
    ax2.grid(True)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x/1000)}k"))

    # 3. Mean Fuel Remaining & Consumption
    ax3 = fig.add_subplot(gs[1, 1])
    fuel_rem = df["mean_fuel_remaining"]
    ax3.plot(timesteps, fuel_rem, color=AERO_COLORS["accent3"], lw=2.2, marker="s", markersize=4, label="Fuel Remaining")
    ax3.set_title("Mean Propellant Conserved at Touchdown", fontweight="bold")
    ax3.set_xlabel("Environment Timesteps")
    ax3.set_ylabel("Remaining Fuel (kg)")
    ax3.set_ylim(0, 105)
    ax3.grid(True)
    ax3.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x/1000)}k"))

    out_path = os.path.join(output_dir, "nominal_learning_curve.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved nominal learning curve plot to: {out_path}")


def plot_sensitivity_analysis(csv_path: str = "results/sensitivity_analysis.csv", output_dir: str = "results"):
    """
    Plot hyperparameter sensitivity comparison across learning rate, discount factor, and entropy coefficient.
    """
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}. Run src/sensitivity_analysis.py first.")
        return

    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    categories = df["category"].unique()

    # Create 3-panel comparison dashboard
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

    titles = {
        "Learning Rate": r"Learning Rate $\alpha$ Sensitivity",
        "Discount Factor": r"Discount Factor $\gamma$ (Planning Horizon)",
        "Exploration-Exploitation (Entropy)": r"Exploration / Entropy Coeff $c_{\text{ent}}$",
    }

    colors_list = [AERO_COLORS["primary"], AERO_COLORS["accent1"], AERO_COLORS["danger"], AERO_COLORS["secondary"]]

    for ax_idx, cat in enumerate(["Learning Rate", "Discount Factor", "Exploration-Exploitation (Entropy)"]):
        ax = axes[ax_idx]
        cat_df = df[df["category"] == cat]
        param_values = cat_df["value"].unique()

        for idx, val in enumerate(param_values):
            sub = cat_df[cat_df["value"] == val].sort_values("timestep")
            color = colors_list[idx % len(colors_list)]
            label_name = f"{sub['parameter'].iloc[0]} = {val}"
            ax.plot(sub["timestep"], sub["mean_reward"], color=color, label=label_name, lw=2.2)
            ax.fill_between(
                sub["timestep"],
                sub["mean_reward"] - 0.5 * sub["std_reward"],
                sub["mean_reward"] + 0.5 * sub["std_reward"],
                color=color,
                alpha=0.15,
            )

        ax.axhline(y=200, color="gray", linestyle=":", lw=1.5, label="Solved (200 pts)")
        ax.axhline(y=0, color="black", linestyle="-", lw=0.8, alpha=0.4)
        ax.set_title(titles.get(cat, cat), fontweight="bold")
        ax.set_xlabel("Environment Timesteps")
        if ax_idx == 0:
            ax.set_ylabel("Mean Evaluation Reward")
        ax.grid(True)
        ax.legend(loc="lower right", framealpha=0.9)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x/1000)}k"))

    plt.suptitle("Hyperparameter Sensitivity Analysis: Policy Convergence and Sample Efficiency", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "sensitivity_comparison_dashboard.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved sensitivity comparison dashboard to: {out_path}")

    # Generate individual dedicated plots for each parameter
    for cat in categories:
        cat_df = df[df["category"] == cat]
        param_name = cat_df["parameter"].iloc[0]
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for idx, val in enumerate(cat_df["value"].unique()):
            sub = cat_df[cat_df["value"] == val].sort_values("timestep")
            color = colors_list[idx % len(colors_list)]
            ax.plot(sub["timestep"], sub["mean_reward"], color=color, label=f"{param_name} = {val}", lw=2.4)
            ax.fill_between(
                sub["timestep"],
                sub["mean_reward"] - sub["std_reward"],
                sub["mean_reward"] + sub["std_reward"],
                color=color,
                alpha=0.18,
            )
        ax.axhline(y=200, color=AERO_COLORS["danger"], linestyle=":", lw=1.8, label="Solved Benchmark")
        ax.set_title(f"Sensitivity Study: {titles.get(cat, cat)}", fontweight="bold")
        ax.set_xlabel("Environment Timesteps")
        ax.set_ylabel("Mean Evaluation Return")
        ax.grid(True)
        ax.legend(loc="lower right", framealpha=0.9)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x/1000)}k"))

        safe_param_name = param_name.replace("/", "_")
        single_path = os.path.join(output_dir, f"sensitivity_{safe_param_name}.png")
        plt.savefig(single_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved individual sensitivity plot to: {single_path}")


def plot_trajectory_profile(model_path: str = "models/best_model.zip", output_dir: str = "results"):
    """
    Run an optimal deterministic landing trajectory with the trained PPO agent
    and plot physical states, continuous controls, mass reduction, and fuel depletion vs time.
    """
    from stable_baselines3 import PPO
    from src.custom_lander import CustomLunarLanderContinuous

    if not os.path.exists(model_path):
        if os.path.exists(model_path.replace(".zip", "")):
            model_path = model_path.replace(".zip", "")
        else:
            print(f"Model checkpoint not found: {model_path}")
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
    dt = 1.0 / 50.0  # 50 FPS

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
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)

    # Subplot 1: Altitude and Horizontal Position
    axes[0, 0].plot(time_steps, positions_y, color=AERO_COLORS["primary"], lw=2.2, label="Altitude $y$")
    axes[0, 0].plot(time_steps, positions_x, color=AERO_COLORS["accent1"], lw=2.0, linestyle="--", label="Horizontal $x$")
    axes[0, 0].set_ylabel("Position (Norm)")
    axes[0, 0].set_title("Lander Flight Trajectory", fontweight="bold")
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    # Subplot 2: Velocities
    axes[0, 1].plot(time_steps, velocities_y, color=AERO_COLORS["danger"], lw=2.2, label="Vertical $v_y$")
    axes[0, 1].plot(time_steps, velocities_x, color=AERO_COLORS["secondary"], lw=2.0, linestyle="--", label="Lateral $v_x$")
    axes[0, 1].set_ylabel("Velocity (Norm)")
    axes[0, 1].set_title("Translational Velocity Profiles", fontweight="bold")
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    # Subplot 3: Orientation Angle
    axes[1, 0].plot(time_steps, angles, color=AERO_COLORS["accent2"], lw=2.2)
    axes[1, 0].axhline(y=0, color="gray", linestyle=":", alpha=0.7)
    axes[1, 0].set_ylabel("Pitch Angle (deg)")
    axes[1, 0].set_title("Vehicle Orientation Angle $\\theta$", fontweight="bold")
    axes[1, 0].grid(True)

    # Subplot 4: Continuous Throttle Commands
    axes[1, 1].plot(time_steps, main_throttles, color=AERO_COLORS["primary"], lw=2.0, label="Main Throttle $u_{\\text{main}}$")
    axes[1, 1].plot(time_steps, side_throttles, color=AERO_COLORS["accent3"], lw=1.8, linestyle="--", label="Side Throttle $u_{\\text{side}}$")
    axes[1, 1].set_ylabel("Command Action")
    axes[1, 1].set_title("Continuous Engine Actuation", fontweight="bold")
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    # Subplot 5: Propellant Depletion
    axes[2, 0].plot(time_steps, fuels, color=AERO_COLORS["accent3"], lw=2.5)
    axes[2, 0].set_xlabel("Mission Time (s)")
    axes[2, 0].set_ylabel("Fuel Remaining (kg)")
    axes[2, 0].set_title("Propellant Mass Depletion", fontweight="bold")
    axes[2, 0].grid(True)

    # Subplot 6: Vehicle Dynamic Mass
    axes[2, 1].plot(time_steps, masses, color=AERO_COLORS["secondary"], lw=2.5)
    axes[2, 1].set_xlabel("Mission Time (s)")
    axes[2, 1].set_ylabel("Vehicle Mass (kg)")
    axes[2, 1].set_title("Dynamic Mass-Varying System $m(t)$", fontweight="bold")
    axes[2, 1].grid(True)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "trajectory_analysis.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved state trajectory profile to: {out_path}")


def main():
    plot_nominal_learning_curve()
    plot_sensitivity_analysis()
    plot_trajectory_profile()


if __name__ == "__main__":
    main()
