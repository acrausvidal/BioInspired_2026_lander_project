import os
import sys
import argparse

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from src.custom_lander import CustomLunarLanderContinuous
from src.callbacks import DetailedEvalCallback


def train(
    max_timesteps: int = 350000,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    ent_coef: float = 0.01,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_envs: int = 4,
    target_reward: float = 190.0,
    target_success: float = 0.85,
    stop_on_convergence: bool = True,
    seed: int = 42,
    save_dir: str = "models",
    results_dir: str = "results",
):
    """
    Train PPO on custom mass-varying Lunar Lander until convergence criteria are met,
    subject to an upper safety limit of max_timesteps.
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print("=" * 75)
    print("TU Delft AE4350 - Bio-inspired Intelligence (Convergence-Criteria Branch)")
    print("Training PPO with Convergence-Based Stopping")
    print("=" * 75)
    print(f"Hyperparameters & Criteria:")
    print(f"  - Max Safety Timesteps:    {max_timesteps:,}")
    print(f"  - Target Return Criterion: {target_reward:.1f} (eval mean)")
    print(f"  - Target Success Rate:     {target_success*100:.1f}%")
    print(f"  - Stop on Convergence:     {stop_on_convergence}")
    print(f"  - Learning Rate:           {learning_rate}")
    print(f"  - Discount Factor:         {gamma}")
    print(f"  - Entropy Coeff:           {ent_coef}")
    print(f"  - Parallel Envs:           {n_envs}")
    print("=" * 75)

    np.random.seed(seed)

    # Vectorized environment
    env = make_vec_env(
        CustomLunarLanderContinuous,
        n_envs=n_envs,
        seed=seed,
        monitor_dir="logs",
    )

    eval_env = Monitor(CustomLunarLanderContinuous())

    csv_log_path = os.path.join(results_dir, "convergence_learning_curve.csv")
    best_model_path = os.path.join(save_dir, "best_model_convergence")

    eval_callback = DetailedEvalCallback(
        eval_env=eval_env,
        eval_freq=10000 // n_envs,
        n_eval_episodes=15,
        log_path=csv_log_path,
        best_model_save_path=best_model_path,
        target_reward_threshold=target_reward,
        target_success_rate=target_success,
        stop_on_convergence=stop_on_convergence,
        deterministic=True,
        verbose=1,
    )

    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=0,
    )

    model.learn(total_timesteps=max_timesteps, callback=eval_callback)

    final_model_path = os.path.join(save_dir, "ppo_convergence_final")
    model.save(final_model_path)

    env.close()
    eval_env.close()
    return eval_callback


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO until convergence")
    parser.add_argument("--max-timesteps", type=int, default=350000, help="Max safety timesteps")
    parser.add_argument("--target-reward", type=float, default=190.0, help="Convergence return threshold")
    parser.add_argument("--target-success", type=float, default=0.85, help="Convergence success rate threshold")
    parser.add_argument("--no-stop", action="store_true", help="Do not stop early on convergence")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--envs", type=int, default=4, help="Parallel environments")
    args = parser.parse_args()

    train(
        max_timesteps=args.max_timesteps,
        target_reward=args.target_reward,
        target_success=args.target_success,
        stop_on_convergence=not args.no_stop,
        learning_rate=args.lr,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        n_envs=args.envs,
    )
