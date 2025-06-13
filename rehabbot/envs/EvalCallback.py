"""
Evaluate periodically the performance of an agent, using a separate test environment. 
It will save the best model if best_model_save_path folder is specified 
and save the evaluations results in a NumPy archive (evaluations.npz) if log_path folder is specified.
"""

import gymnasium as gym
import rehabbot

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

# Separate evaluation env
eval_env = gym.make("rehabbot/rehab-reach-v0")
# Use deterministic actions for evaluation
eval_callback = EvalCallback(eval_env, best_model_save_path="./checkpoint/",
                             log_path="./checkpoint/", eval_freq=5000,
                             deterministic=True, render=False)

model = PPO("MlpPolicy", "rehabbot/rehab-reach-v0")
model.learn(50000, callback=eval_callback)