import os
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv


class DetailedEvalCallback(BaseCallback):
    """
    Callback for evaluating policy at regular intervals during training,
    logging performance metrics (Reward, Success Rate, Propellant, Mass, Length)
    against BOTH environment timesteps and episode counts.
    Supports early stopping upon meeting aerospace convergence criteria.
    """

    def __init__(
        self,
        eval_env,
        eval_freq: int = 10000,
        n_eval_episodes: int = 15,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        target_reward_threshold: Optional[float] = None,
        target_success_rate: float = 0.85,
        stop_on_convergence: bool = False,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.log_path = log_path
        self.best_model_save_path = best_model_save_path
        self.deterministic = deterministic
        self.target_reward_threshold = target_reward_threshold
        self.target_success_rate = target_success_rate
        self.stop_on_convergence = stop_on_convergence
        
        self.best_mean_reward = -np.inf
        self.evaluations_results: List[Dict[str, Any]] = []
        self.total_episodes_seen = 0
        self.is_converged = False
        self.convergence_timestep = None
        self.convergence_episode = None

    def _init_callback(self) -> None:
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        if self.best_model_save_path is not None:
            os.makedirs(os.path.dirname(self.best_model_save_path), exist_ok=True)

    def _on_step(self) -> bool:
        # Approximate episode count from training environment Monitor infos if available
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.total_episodes_seen += 1

        if self.n_calls % self.eval_freq == 0:
            rewards = []
            episode_lengths = []
            fuel_remaining_list = []
            fuel_consumed_list = []
            success_list = []
            out_of_fuel_list = []
            crash_list = []

            for _ in range(self.n_eval_episodes):
                obs, info = self.eval_env.reset()
                done = False
                ep_reward = 0.0
                ep_len = 0
                last_info = info

                while not done:
                    action, _ = self.model.predict(obs, deterministic=self.deterministic)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    ep_reward += reward
                    ep_len += 1
                    done = terminated or truncated
                    last_info = info

                rewards.append(ep_reward)
                episode_lengths.append(ep_len)
                fuel_rem = last_info.get("fuel_remaining", 0.0)
                fuel_cons = last_info.get("fuel_consumed", 100.0 - fuel_rem)
                fuel_remaining_list.append(fuel_rem)
                fuel_consumed_list.append(fuel_cons)
                
                is_success = last_info.get("is_safe_landing", False) or ep_reward > 100.0
                is_oof = last_info.get("out_of_fuel", False) or (fuel_rem <= 0.0)
                is_crash = (not is_success) and (not is_oof) and (ep_reward < 0.0)

                success_list.append(float(is_success))
                out_of_fuel_list.append(float(is_oof))
                crash_list.append(float(is_crash))

            mean_reward = float(np.mean(rewards))
            std_reward = float(np.std(rewards))
            mean_len = float(np.mean(episode_lengths))
            mean_fuel_rem = float(np.mean(fuel_remaining_list))
            mean_fuel_cons = float(np.mean(fuel_consumed_list))
            success_rate = float(np.mean(success_list))
            oof_rate = float(np.mean(out_of_fuel_list))
            crash_rate = float(np.mean(crash_list))

            # Fraction of evaluation episodes meeting high-performance criterion (reward >= 190)
            high_reward_fraction = float(np.mean([1.0 if r >= 190.0 else 0.0 for r in rewards]))

            log_entry = {
                "timestep": int(self.num_timesteps),
                "episodes_approx": int(self.total_episodes_seen),
                "mean_reward": mean_reward,
                "std_reward": std_reward,
                "min_reward": float(np.min(rewards)),
                "max_reward": float(np.max(rewards)),
                "high_reward_pct": high_reward_fraction,
                "mean_episode_length": mean_len,
                "mean_fuel_remaining": mean_fuel_rem,
                "mean_fuel_consumed": mean_fuel_cons,
                "success_rate": success_rate,
                "out_of_fuel_rate": oof_rate,
                "crash_rate": crash_rate,
            }
            self.evaluations_results.append(log_entry)

            if self.verbose > 0:
                print(
                    f"[{self.num_timesteps:7d} steps | ~{self.total_episodes_seen:4d} eps] "
                    f"Mean Reward: {mean_reward:6.2f} +/- {std_reward:5.2f} | "
                    f"Success Rate: {success_rate*100:5.1f}% | "
                    f"Fuel Left: {mean_fuel_rem:5.1f} kg"
                )

            # Save best checkpoint
            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print(f"  --> New best mean reward: {mean_reward:.2f} (prev: {self.best_mean_reward:.2f}). Saving checkpoint...")
                self.best_mean_reward = mean_reward
                if self.best_model_save_path is not None:
                    self.model.save(self.best_model_save_path)

            # Export to CSV
            if self.log_path is not None:
                df = pd.DataFrame(self.evaluations_results)
                df.to_csv(self.log_path, index=False)

            # Check Early Stopping Convergence Criteria
            if self.target_reward_threshold is not None:
                # Criteria: Mean reward >= threshold AND Success rate >= target_success_rate
                if mean_reward >= self.target_reward_threshold and success_rate >= self.target_success_rate:
                    self.is_converged = True
                    self.convergence_timestep = int(self.num_timesteps)
                    self.convergence_episode = int(self.total_episodes_seen)
                    if self.verbose > 0:
                        print("\n" + "*" * 75)
                        print(f"*** CONVERGENCE CRITERIA MET AT TIMESTEP {self.num_timesteps} (~{self.total_episodes_seen} EPISODES) ***")
                        print(f"*** Mean Return: {mean_reward:.2f} >= {self.target_reward_threshold:.2f} | Success Rate: {success_rate*100:.1f}% >= {self.target_success_rate*100:.1f}% ***")
                        print("*" * 75 + "\n")
                    
                    if self.stop_on_convergence:
                        if self.verbose > 0:
                            print("Stopping training early as convergence has been verified.")
                        return False

        return True

    def _on_training_end(self) -> None:
        if self.log_path is not None and self.evaluations_results:
            df = pd.DataFrame(self.evaluations_results)
            df.to_csv(self.log_path, index=False)
            if self.verbose > 0:
                print(f"Evaluation logs successfully saved to {self.log_path}")
