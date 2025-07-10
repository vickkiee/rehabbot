import gymnasium as gym
import rehabbot

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

# Separate evaluation env
eval_env = gym.make("rehabbot/rehab-reach-v0")

# Stop training when the model reaches the reward threshold
callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=48, verbose=1)
eval_callback = EvalCallback(eval_env, callback_on_new_best=callback_on_best, verbose=1)






model = PPO("MlpPolicy", "rehabbot/rehab-reach-v0", verbose=1)
# Almost infinite number of timesteps, but the training will stop early as soon as the reward threshold is reached
model.learn(int(1e10), callback=eval_callback)