import gymnasium as gym
import rehabbot

def test_env_instantiation():
    env = gym.make("rehabbot/rehab-reach-v0")
    obs, _ = env.reset(seed=0)
    assert obs is not None
    env.close()


def test_env_step():
    env = gym.make("rehabbot/rehab-reach-v0")
    obs, _ = env.reset(seed=0)

    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)

    assert reward is not None
    env.close()