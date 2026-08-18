import os
import sys

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from stable_baselines3 import PPO
from src.custom_lander import CustomLunarLanderContinuous


def main():
    """
    Demo script: Runs the custom mass-varying lunar lander environment
    using either the trained best PPO model (if available) or random actions with visual rendering.
    """
    model_path = "models/best_model.zip"
    use_trained_model = os.path.exists(model_path)

    env = CustomLunarLanderContinuous(render_mode="human")

    if use_trained_model:
        print(f"Loading trained PPO policy from {model_path}...")
        model = PPO.load(model_path)
    else:
        print("No trained checkpoint found. Running untrained demonstration...")
        model = None

    obs, info = env.reset()
    total_reward = 0.0

    print("Running demonstration (press Ctrl+C or close window to stop)...")
    try:
        for step in range(1000):
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                print(f"Episode finished! Cumulative Reward: {total_reward:.2f} | Final Info: {info}")
                obs, info = env.reset()
                total_reward = 0.0
    except KeyboardInterrupt:
        print("\nDemonstration stopped by user.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
