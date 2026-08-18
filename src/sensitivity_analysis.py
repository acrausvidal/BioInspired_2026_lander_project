import os
import sys
import time
from typing import Dict, Any, List

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from src.custom_lander import CustomLunarLanderContinuous
from src.callbacks import DetailedEvalCallback


def run_experiment(
    exp_category: str,
    param_name: str,
    param_value: Any,
    ppo_kwargs: Dict[str, Any],
    total_timesteps: int = 200000,
    eval_freq_steps: int = 10000,
    n_envs: int = 4,
    seed: int = 42,
    results_dir: str = "results",
) -> pd.DataFrame:
    """
    Run a single training experiment with specified hyperparameters and collect evaluation logs.
    """
    print("\n" + "=" * 70)
    print(f"Running Experiment: {exp_category} | {param_name} = {param_value}")
    print("=" * 70)

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = make_vec_env(
        CustomLunarLanderContinuous,
        n_envs=n_envs,
        seed=seed,
        monitor_dir="logs",
    )
    eval_env = Monitor(CustomLunarLanderContinuous())

    # Temporary CSV for this run
    run_tag = f"{exp_category}_{param_name}_{param_value}".replace(".", "p")
    run_csv_path = os.path.join(results_dir, f"temp_{run_tag}.csv")

    eval_callback = DetailedEvalCallback(
        eval_env=eval_env,
        eval_freq=eval_freq_steps // n_envs,
        n_eval_episodes=15,
        log_path=run_csv_path,
        best_model_save_path=os.path.join("models", f"best_{run_tag}"),
        deterministic=True,
        verbose=1,
    )

    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=0,
        **ppo_kwargs,
    )

    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.1f} seconds.")

    env.close()
    eval_env.close()

    # Load results and attach metadata
    if os.path.exists(run_csv_path):
        df = pd.read_csv(run_csv_path)
        os.remove(run_csv_path)
    else:
        df = pd.DataFrame(eval_callback.evaluations_results)

    df["category"] = exp_category
    df["parameter"] = param_name
    df["value"] = str(param_value)
    df["label"] = f"{param_name} = {param_value}"

    return df


def main():
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    timesteps_per_run = 150000  # 150k steps per run is optimal to benchmark convergence speed & stability
    n_envs = 4
    seed = 42

    all_results = []

    # -------------------------------------------------------------
    # 1. Learning Rate Sensitivity Study (\alpha)
    # -------------------------------------------------------------
    lr_values = [1e-4, 3e-4, 1e-3]
    for lr in lr_values:
        ppo_kwargs = {
            "learning_rate": lr,
            "gamma": 0.99,
            "ent_coef": 0.01,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
        }
        df = run_experiment(
            exp_category="Learning Rate",
            param_name="learning_rate",
            param_value=lr,
            ppo_kwargs=ppo_kwargs,
            total_timesteps=timesteps_per_run,
            n_envs=n_envs,
            seed=seed,
        )
        all_results.append(df)

    # -------------------------------------------------------------
    # 2. Discount Factor Sensitivity Study (\gamma) - Planning Horizon
    # -------------------------------------------------------------
    gamma_values = [0.95, 0.99, 0.999]
    for gamma in gamma_values:
        ppo_kwargs = {
            "learning_rate": 3e-4,
            "gamma": gamma,
            "ent_coef": 0.01,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
        }
        df = run_experiment(
            exp_category="Discount Factor",
            param_name="gamma",
            param_value=gamma,
            ppo_kwargs=ppo_kwargs,
            total_timesteps=timesteps_per_run,
            n_envs=n_envs,
            seed=seed,
        )
        all_results.append(df)

    # -------------------------------------------------------------
    # 3. Exploration vs. Exploitation Study (Entropy Coefficient c_{ent})
    # -------------------------------------------------------------
    ent_values = [0.0, 0.01, 0.05]
    for ent in ent_values:
        ppo_kwargs = {
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "ent_coef": ent,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
        }
        df = run_experiment(
            exp_category="Exploration-Exploitation (Entropy)",
            param_name="ent_coef",
            param_value=ent,
            ppo_kwargs=ppo_kwargs,
            total_timesteps=timesteps_per_run,
            n_envs=n_envs,
            seed=seed,
        )
        all_results.append(df)

    # Combine all results into master CSV
    combined_df = pd.concat(all_results, ignore_index=True)
    master_csv_path = os.path.join(results_dir, "sensitivity_analysis.csv")
    combined_df.to_csv(master_csv_path, index=False)
    print("\n" + "=" * 70)
    print(f"All sensitivity experiments completed successfully!")
    print(f"Master dataset saved to: {master_csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
