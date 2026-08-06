import os
import gymnasium as gym
from stable_baselines3 import PPO


def train():
    # Ensure directory structure exists
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # 1. Initialize LunarLanderContinuous-v3 environment without rendering
    env = gym.make("LunarLanderContinuous-v3")

    # 2. Create PPO model with MlpPolicy and tensorboard logging
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")

    # 3. Train for 250,000 timesteps
    print("Starting training for 250,000 timesteps...")
    model.learn(total_timesteps=250000)

    # 4. Save model to ./models/ppo_baseline
    save_path = "./models/ppo_baseline"
    model.save(save_path)
    print(f"Model saved successfully to {save_path}")

    env.close()


if __name__ == "__main__":
    train()
