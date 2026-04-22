import gymnasium as gym
import rehabbot  # registers env IDs

env = gym.make("rehabbot/rehab-reach-v0", render_mode="human")
obs, info = env.reset(seed=0)

done = False

while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()