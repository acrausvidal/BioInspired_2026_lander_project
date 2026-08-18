import os
import sys
import argparse

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

from src.custom_lander import CustomLunarLanderContinuous


def evaluate_policy(
    model_path: str = "models/best_model",
    num_episodes: int = 20,
    render: bool = False,
    seed: int = 42,
):
    """
    Evaluate a trained PPO policy on the Custom Lunar Lander environment.
    """
    # Check model path
    if not os.path.exists(model_path) and not os.path.exists(f"{model_path}.zip"):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Please run training first.")

    render_mode = "human" if render else None
    env = CustomLunarLanderContinuous(render_mode=render_mode)
    print(f"Loading trained policy from: {model_path}...")
    model = PPO.load(model_path)

    rewards = []
    episode_lengths = []
    fuels_remaining = []
    fuels_consumed = []
    successes = []

    print("=" * 65)
    print(f"Starting Evaluation over {num_episodes} Episodes (Deterministic Policy)")
    print("=" * 65)

    for ep in range(1, num_episodes + 1):
        obs, info = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0
        steps = 0
        last_info = info

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1
            done = terminated or truncated
            last_info = info

        fuel_rem = last_info.get("fuel_remaining", 0.0)
        fuel_cons = last_info.get("fuel_consumed", 100.0 - fuel_rem)
        is_success = last_info.get("is_safe_landing", False) or ep_reward > 100.0

        rewards.append(ep_reward)
        episode_lengths.append(steps)
        fuels_remaining.append(fuel_rem)
        fuels_consumed.append(fuel_cons)
        successes.append(is_success)

        status_str = "SAFE TOUCHDOWN" if is_success else "CRASH / OOF"
        print(
            f"Episode {ep:2d}/{num_episodes:2d} | "
            f"Reward: {ep_reward:6.2f} | "
            f"Steps: {steps:3d} | "
            f"Fuel Rem: {fuel_rem:5.1f} kg | "
            f"Mass: {last_info.get('lander_mass', 0):.2f} kg | "
            f"Status: {status_str}"
        )

    env.close()

    print("=" * 65)
    print("EVALUATION SUMMARY STATISTICS:")
    print(f"  - Mean Return:          {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"  - Max / Min Return:     {np.max(rewards):.2f} / {np.min(rewards):.2f}")
    print(f"  - Landing Success Rate: {np.mean(successes)*100:.1f}%")
    print(f"  - Mean Propellant Left: {np.mean(fuels_remaining):.2f} kg ({np.mean(fuels_remaining)/100*100:.1f}%)")
    print(f"  - Mean Propellant Burn: {np.mean(fuels_consumed):.2f} kg")
    print(f"  - Mean Flight Duration: {np.mean(episode_lengths)/50.0:.2f} seconds ({np.mean(episode_lengths):.0f} steps)")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Trained Lunar Lander Agent")
    parser.add_argument("--model", type=str, default="models/best_model", help="Path to trained model")
    parser.add_argument("--episodes", type=int, default=20, help="Number of test episodes")
    parser.add_argument("--render", action="store_true", help="Render simulation to display")
    parser.add_argument("--seed", type=int, default=42, help="Evaluation seed")
    args = parser.parse_args()

    evaluate_policy(
        model_path=args.model,
        num_episodes=args.episodes,
        render=args.render,
        seed=args.seed,
    )
