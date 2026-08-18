import os
import sys
import time
from typing import Dict, Any, List
import pandas as pd
import numpy as np

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from src.custom_lander import CustomLunarLanderContinuous
from src.callbacks import DetailedEvalCallback


def run_convergence_experiment(
    exp_category: str,
    param_name: str,
    param_value: Any,
    ppo_kwargs: Dict[str, Any],
    max_safety_timesteps: int = 250000,
    eval_freq_steps: int = 10000,
    target_reward: float = 190.0,
    target_success: float = 0.85,
    n_envs: int = 4,
    seed: int = 42,
    results_dir: str = "results",
) -> Dict[str, Any]:
    """
    Run training until convergence criteria are met (or max safety timesteps reached).
    """
    print("\n" + "=" * 75)
    print(f"Convergence Run: {exp_category} | {param_name} = {param_value}")
    print(f"Target Criterion: Mean Return >= {target_reward} & Success Rate >= {target_success*100:.0f}%")
    print("=" * 75)

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    env = make_vec_env(
        CustomLunarLanderContinuous,
        n_envs=n_envs,
        seed=seed,
        monitor_dir="logs",
    )
    eval_env = Monitor(CustomLunarLanderContinuous())

    run_tag = f"conv_{exp_category}_{param_name}_{param_value}".replace(".", "p").replace(" ", "_")
    run_csv_path = os.path.join(results_dir, f"temp_{run_tag}.csv")

    eval_callback = DetailedEvalCallback(
        eval_env=eval_env,
        eval_freq=eval_freq_steps // n_envs,
        n_eval_episodes=15,
        log_path=run_csv_path,
        best_model_save_path=os.path.join("models", f"best_{run_tag}"),
        target_reward_threshold=target_reward,
        target_success_rate=target_success,
        stop_on_convergence=True,
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
    model.learn(total_timesteps=max_safety_timesteps, callback=eval_callback)
    elapsed = time.time() - start_time

    env.close()
    eval_env.close()

    if os.path.exists(run_csv_path):
        df = pd.read_csv(run_csv_path)
        os.remove(run_csv_path)
    else:
        df = pd.DataFrame(eval_callback.evaluations_results)

    df["category"] = exp_category
    df["parameter"] = param_name
    df["value"] = str(param_value)
    df["label"] = f"{param_name} = {param_value}"

    last_eval = df.iloc[-1] if not df.empty else {}

    summary = {
        "category": exp_category,
        "parameter": param_name,
        "value": str(param_value),
        "converged": eval_callback.is_converged,
        "steps_to_converge": eval_callback.convergence_timestep if eval_callback.is_converged else max_safety_timesteps,
        "episodes_to_converge": eval_callback.convergence_episode if eval_callback.is_converged else int(df["episodes_approx"].max() if "episodes_approx" in df else 0),
        "final_mean_reward": last_eval.get("mean_reward", np.nan),
        "final_success_rate": last_eval.get("success_rate", np.nan),
        "final_fuel_remaining": last_eval.get("mean_fuel_remaining", np.nan),
        "wall_clock_time_s": elapsed,
    }

    return {"df": df, "summary": summary}


def main():
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    max_safety_steps = 200000
    n_envs = 4
    seed = 42

    all_dfs = []
    summaries = []

    # 1. Learning Rate
    for lr in [1e-4, 3e-4, 1e-3]:
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
        res = run_convergence_experiment(
            exp_category="Learning Rate",
            param_name="learning_rate",
            param_value=lr,
            ppo_kwargs=ppo_kwargs,
            max_safety_timesteps=max_safety_steps,
            n_envs=n_envs,
            seed=seed,
        )
        all_dfs.append(res["df"])
        summaries.append(res["summary"])

    # 2. Discount Factor
    for gamma in [0.95, 0.99, 0.999]:
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
        res = run_convergence_experiment(
            exp_category="Discount Factor",
            param_name="gamma",
            param_value=gamma,
            ppo_kwargs=ppo_kwargs,
            max_safety_timesteps=max_safety_steps,
            n_envs=n_envs,
            seed=seed,
        )
        all_dfs.append(res["df"])
        summaries.append(res["summary"])

    # 3. Entropy Coefficient
    for ent in [0.0, 0.01, 0.05]:
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
        res = run_convergence_experiment(
            exp_category="Exploration-Exploitation (Entropy)",
            param_name="ent_coef",
            param_value=ent,
            ppo_kwargs=ppo_kwargs,
            max_safety_timesteps=max_safety_steps,
            n_envs=n_envs,
            seed=seed,
        )
        all_dfs.append(res["df"])
        summaries.append(res["summary"])

    # Combine trajectories and summaries
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.to_csv(os.path.join(results_dir, "convergence_sensitivity_analysis.csv"), index=False)

    summary_df = pd.DataFrame(summaries)
    summary_csv = os.path.join(results_dir, "convergence_summary_table.csv")
    summary_df.to_csv(summary_csv, index=False)

    print("\n" + "=" * 75)
    print("CONVERGENCE BENCHMARK SUMMARY TABLE:")
    print("=" * 75)
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary table to: {summary_csv}")


if __name__ == "__main__":
    main()
