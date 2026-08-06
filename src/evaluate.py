import os
import gymnasium as gym
from stable_baselines3 import PPO


def evaluate():
    model_path = "./models/ppo_baseline"
    if not os.path.exists(f"{model_path}.zip") and not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Saved model not found at {model_path}. Please run src/train.py first."
        )

    # 1. Initialize LunarLanderContinuous-v3 environment with human rendering
    env = gym.make("LunarLanderContinuous-v3", render_mode="human")

    # 2. Load the trained PPO model
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)

    # 3. Standard evaluation loop for 5 full episodes
    num_episodes = 5
    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        print(f"Episode {episode}: Cumulative Reward = {episode_reward:.2f}")

    env.close()


if __name__ == "__main__":
    evaluate()
