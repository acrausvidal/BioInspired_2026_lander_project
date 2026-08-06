import gymnasium as gym
from stable_baselines3 import PPO


def main():
    # 1. Initialize LunarLanderContinuous-v3 environment from Gymnasium with human rendering
    env = gym.make("LunarLanderContinuous-v3", render_mode="human")

    # 2. Set up basic PPO model with default hyperparameters
    model = PPO("MlpPolicy", env, verbose=1)

    # 3. Run a simple loop for 1000 steps using untrained PPO predictions
    obs, info = env.reset()
    for _ in range(1000):
        # Predict action using untrained model
        action, _states = model.predict(obs)
        
        # Take step in environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Reset environment if episode finished
        if terminated or truncated:
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
