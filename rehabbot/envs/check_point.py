from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, ProgressBarCallback

import time
import rehabbot

import gymnasium as gym
from gymnasium.core import ObsType, ActType, SupportsFloat, RenderFrame
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.wrappers import TimeLimit

# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(
  save_freq=20000,
  save_path="./checkpoint/",
  name_prefix="rl_model",
  save_replay_buffer=True,
  save_vecnormalize=True,
)

model = PPO("MlpPolicy", "rehabbot/rehab-reach-v0")
model.learn(50000, callback=checkpoint_callback)