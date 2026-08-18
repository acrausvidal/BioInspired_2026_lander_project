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
    total_timesteps: int = 300000,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    ent_coef: float = 0.01,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_envs: int = 4,
    seed: int = 42,
    save_dir: str = "models",
    results_dir: str = "results",
):
    """
    Train a nominal PPO agent on the mass-varying continuous Lunar Lander environment.
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print("=" * 70)
    print("TU Delft AE4350 - Bio-inspired Intelligence")
    print("Training PPO Agent on Custom Mass-Varying Lunar Lander")
    print("=" * 70)
    print(f"Hyperparameters:")
    print(f"  - Total Timesteps:  {total_timesteps:,}")
    print(f"  - Learning Rate:    {learning_rate}")
    print(f"  - Discount Factor:  {gamma}")
    print(f"  - Entropy Coeff:    {ent_coef}")
    print(f"  - Batch Size:       {batch_size}")
    print(f"  - n_steps (rollout):{n_steps}")
    print(f"  - Parallel Envs:    {n_envs}")
    print(f"  - Random Seed:      {seed}")
    print("=" * 70)

    # Set random seeds for reproducibility
    np.random.seed(seed)

    # 1. Create vectorized training environment
    env = make_vec_env(
        CustomLunarLanderContinuous,
        n_envs=n_envs,
        seed=seed,
        monitor_dir="logs",
    )

    # 2. Create single evaluation environment
    eval_env = Monitor(CustomLunarLanderContinuous())

    # 3. Set up evaluation and metric logging callback
    csv_log_path = os.path.join(results_dir, "nominal_learning_curve.csv")
    best_model_path = os.path.join(save_dir, "best_model")

    eval_callback = DetailedEvalCallback(
        eval_env=eval_env,
        eval_freq=10000 // n_envs,  # Evaluate every 10,000 global timesteps
        n_eval_episodes=15,
        log_path=csv_log_path,
        best_model_save_path=best_model_path,
        deterministic=True,
        verbose=1,
    )

    # 4. Initialize PPO agent
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
        tensorboard_log="./logs/ppo_tb/",
        seed=seed,
        verbose=0,
    )

    # 5. Train agent
    print("Starting nominal training run...")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    # 6. Save final model
    final_model_path = os.path.join(save_dir, "ppo_nominal_final")
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}.zip")
    print(f"Best checkpoint saved to: {best_model_path}.zip")
    print(f"Training metrics logged to: {csv_log_path}")

    # Cleanup
    env.close()
    eval_env.close()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on Custom Lunar Lander")
    parser.add_argument("--timesteps", type=int, default=300000, help="Total training timesteps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train(
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        n_envs=args.envs,
        seed=args.seed,
    )
