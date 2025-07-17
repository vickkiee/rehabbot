import gymnasium as gym
import numpy as np
import rehabbot

from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, DQN, SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import BaseCallback




class EpisodeInfoCallback(BaseCallback):
    """
    Callback to extract and store the length (number of steps) of each episode.
    The Monitor wrapper includes, in the `info` dict under the key 'episode',
    a dict containing, among other values, 'l' (the episode length in steps).
    """
    def __init__(self, verbose=0):
        super(EpisodeInfoCallback, self).__init__(verbose)
        self.episode_lengths = []  # List to accumulate episode lengths
        self.last_values = 0

    def _on_step(self) -> bool:
        # Get the "infos" list from the rollouts
        infos = self.locals.get("infos", [])
        for info in infos:
            # If the info dict has an "episode" key, an episode has ended
            if "episode" in info:
                self.episode_lengths.append(info["episode"]["l"])
        return True

    def get_mean_episode_length(self):
        # Use the most recent 2000 lengths for averaging if available
        new_mean = self.episode_lengths[-2000:] if self.episode_lengths else []
        if self.last_values == 0:
            # Initialize last_values with the most recent 100
            self.last_values = self.episode_lengths[-100:]
        # Compute the running mean between new and last values
        self.mean_episode_length = np.mean(new_mean + self.last_values)
        self.last_values = self.episode_lengths[-100:]
        return self.mean_episode_length

    def reset_episodes(self):
        """Reset episode statistics so only recent episodes are considered."""
        self.episode_lengths = []


def train_agents_with_eval(total_timesteps, eval_interval, num_eval_episodes):
    """
    Train a PPO and SAC agents (or whichever you need) inside the training env and, every `eval_interval` timesteps, evaluate the current policy by 
    simulating `num_eval_episodes` in the evaluation environment (CartPole-v1 in this case, you can use whichever you want).

    You can add as many agent evaluations as you like, and later plot them in order to compare them in a single image.

    Metrics are collected and plotted to 'evaluation_reward_comparison.png' and 'training_reward_comparison.png'. You can create your own cowplot if
    you desire to have them all in one single image.
    """
    # ----------------------------
    # 1) Train agent 1
    # ----------------------------
    env = gym.make("rehabbot/rehab-reach-v0")
    env = Monitor(env)
    agent_1 = SAC("MlpPolicy", env, verbose=0)
    episode_callback = EpisodeInfoCallback()

    timesteps_record = []
    agent_1_training_rewards = []
    agent_1_eval_rewards = []

    total_evaluated = 0
    while total_evaluated < total_timesteps:
        # Train for eval_interval steps
        agent_1.learn(total_timesteps=eval_interval, reset_num_timesteps=False, callback=episode_callback)
        mean_len = episode_callback.get_mean_episode_length()
        total_evaluated += eval_interval

        timesteps_record.append(total_evaluated)
        agent_1_training_rewards.append(mean_len)

        # --- Evaluate the trained agent for training steps range stated ---
        eval_rewards = []
        eval_env = gym.make("rehabbot/rehab-reach-v0")
        for _ in range(num_eval_episodes):
            ep_reward = 0
            state, _ = eval_env.reset()
            done = False
            while not done:
                action, _ = agent_1.predict(state)
                state, reward, done, truncated, info = eval_env.step(action)
                ep_reward += reward
                if truncated:
                    done = True
            eval_rewards.append(ep_reward)
        agent_1_eval_rewards.append(np.mean(eval_rewards))
        eval_env.close()

        print(f"Total Steps [SAC Agent]: {total_evaluated} -> " f"Training Reward: {mean_len:.2f}, " f"Eval (2 episodes) Reward: {agent_1_eval_rewards[-1]:.2f}")
        
    # ----------------------------
    # 2) Train agent 2
    # ----------------------------
    env = gym.make("rehabbot/rehab-reach-v0")   # You can remove this one if you want
    env = Monitor(env)  
    agent_2 = PPO("MlpPolicy", env, verbose=0)
    episode_callback = EpisodeInfoCallback()

    timesteps_record = []
    agent_2_training_rewards = []
    agent_2_eval_rewards = []

    total_evaluated = 0
    while total_evaluated < total_timesteps:
        # Train for eval_interval steps
        agent_2.learn(total_timesteps=eval_interval, reset_num_timesteps=False, callback=episode_callback)
        mean_len = episode_callback.get_mean_episode_length()
        total_evaluated += eval_interval

        timesteps_record.append(total_evaluated)
        agent_2_training_rewards.append(mean_len)

        # --- Evaluate the trained agent for training steps range stated ---
        eval_rewards = []
        eval_env = gym.make("rehabbot/rehab-reach-v0")
        for _ in range(num_eval_episodes):
            ep_reward = 0
            state, _ = eval_env.reset()
            done = False
            while not done:
                action, _ = agent_2.predict(state)
                state, reward, done, truncated, info = eval_env.step(action)
                if truncated:
                    done = True
                ep_reward += reward
            eval_rewards.append(ep_reward)
        agent_2_eval_rewards.append(np.mean(eval_rewards))
        eval_env.close()

        print(f"Total Steps [PPO Agent]: {total_evaluated} -> " f"Training Reward: {mean_len:.2f}, " f"Eval (2 episodes) Reward: {agent_2_eval_rewards[-1]:.2f}")

    # ----------------------------
    # 3) Train agent 3
    # ----------------------------
    env = gym.make("rehabbot/rehab-reach-v0")   # You can remove this one if you want
    env = Monitor(env)  
    agent_3 = TD3("MlpPolicy", env, verbose=0)
    episode_callback = EpisodeInfoCallback()

    timesteps_record = []
    agent_3_training_rewards = []
    agent_3_eval_rewards = []

    total_evaluated = 0
    while total_evaluated < total_timesteps:
        # Train for eval_interval steps
        agent_3.learn(total_timesteps=eval_interval, reset_num_timesteps=False, callback=episode_callback)
        mean_len = episode_callback.get_mean_episode_length()
        total_evaluated += eval_interval

        timesteps_record.append(total_evaluated)
        agent_3_training_rewards.append(mean_len)

        # --- Evaluate the trained agent for training steps range stated ---
        eval_rewards = []
        eval_env = gym.make("rehabbot/rehab-reach-v0")
        for _ in range(num_eval_episodes):
            ep_reward = 0
            state, _ = eval_env.reset()
            done = False
            while not done:
                action, _ = agent_3.predict(state)
                state, reward, done, truncated, info = eval_env.step(action)
                if truncated:
                    done = True
                ep_reward += reward
            eval_rewards.append(ep_reward)
        agent_3_eval_rewards.append(np.mean(eval_rewards))
        eval_env.close()

        print(f"Total Steps [TD3 Agent]: {total_evaluated} -> " f"Training Reward: {mean_len:.2f}, " f"Eval (2 episodes) Reward: {agent_3_eval_rewards[-1]:.2f}")
        
    return timesteps_record, agent_1_training_rewards, agent_1_eval_rewards, agent_2_training_rewards, agent_2_eval_rewards, agent_3_training_rewards, agent_3_eval_rewards


# =====================================
# Main function
# =====================================
def main():
    all_agent_1_training_rewards = []
    all_agent_1_evaluation_rewards = []
    all_agent_2_training_rewards = []
    all_agent_2_evaluation_rewards = []
    all_agent_3_training_rewards = []
    all_agent_3_evaluation_rewards = []
    rollouts = 10   # In order to obtain statistical significance (when I use World Models, maybe you don't need it)

    for i in range(rollouts):
        print(f"=== Rollout {i+1}/{rollouts} ===")
        #print("\nTraining SAC agent in RehabBot...")
        timesteps, agent_1_train_rewards, agent_1_eval_rewards, agent_2_train_rewards, agent_2_eval_rewards, agent_3_train_rewards, agent_3_eval_rewards = train_agents_with_eval(total_timesteps=1500000, eval_interval=10000, num_eval_episodes=50)

        all_agent_1_training_rewards.append(agent_1_train_rewards)
        all_agent_1_evaluation_rewards.append(agent_1_eval_rewards)
        all_agent_2_training_rewards.append(agent_2_train_rewards)
        all_agent_2_evaluation_rewards.append(agent_2_eval_rewards)
        all_agent_3_training_rewards.append(agent_3_train_rewards)
        all_agent_3_evaluation_rewards.append(agent_3_eval_rewards)

        if i > 0:
            # Compute mean and 95% confidence interval
            def compute_mean_ci(data, z=1.96):
                means = np.mean(data, axis=0)
                sem = np.std(data, axis=0) / np.sqrt(data.shape[0])
                ci = z * sem
                return means, ci

            agent_1_train_mean, agent_1_train_ci = compute_mean_ci(np.array(all_agent_1_training_rewards))
            agent_1_eval_mean, agent_1_eval_ci = compute_mean_ci(np.array(all_agent_1_evaluation_rewards))
            agent_2_train_mean, agent_2_train_ci = compute_mean_ci(np.array(all_agent_2_training_rewards))
            agent_2_eval_mean, agent_2_eval_ci = compute_mean_ci(np.array(all_agent_2_evaluation_rewards))
            agent_3_train_mean, agent_3_train_ci = compute_mean_ci(np.array(all_agent_3_training_rewards))
            agent_3_eval_mean, agent_3_eval_ci = compute_mean_ci(np.array(all_agent_3_evaluation_rewards))

            # Plot evaluation rewards
            plt.figure(figsize=(10, 6))
            plt.plot(timesteps, agent_1_eval_mean, color="orange", label="SAC Evaluation Mean Reward (RH)")
            plt.fill_between(timesteps, agent_1_eval_mean - agent_1_eval_ci, agent_1_eval_mean + agent_1_eval_ci, color="orange", alpha=0.2)
            plt.plot(timesteps, agent_2_eval_mean, color="blue", label="PPO Evaluation Mean Reward (RH)")
            plt.fill_between(timesteps, agent_2_eval_mean - agent_2_eval_ci, agent_2_eval_mean + agent_2_eval_ci, color="blue", alpha=0.2)
            plt.plot(timesteps, agent_3_eval_mean, color="green", label="TD3 Evaluation Mean Reward (RH)")
            plt.fill_between(timesteps, agent_3_eval_mean - agent_3_eval_ci, agent_3_eval_mean + agent_3_eval_ci, color="green", alpha=0.2)
            plt.xlabel("Training Timesteps")
            plt.ylabel("Mean Episode Reward")
            plt.title(f"RehabBot Mean Reward - Evaluation ({i+1} runs)")
            plt.legend()
            plt.grid(True)
            plt.savefig("evaluation_reward_comparison.png")
            plt.show()
            plt.close()

            # Plot training rewards
            plt.figure(figsize=(10, 6))
            plt.plot(timesteps, agent_1_train_mean, color="orange", label="SAC Training Reward (RH)")
            plt.fill_between(timesteps, agent_1_train_mean - agent_1_train_ci, agent_1_train_mean + agent_1_train_ci, color="orange", alpha=0.2)
            plt.plot(timesteps, agent_2_train_mean, color="blue", label="PPO Training Reward (RH)")
            plt.fill_between(timesteps, agent_2_train_mean - agent_2_train_ci, agent_2_train_mean + agent_2_train_ci, color="blue", alpha=0.2)
            plt.plot(timesteps, agent_3_train_mean, color="green", label="TD3 Training Reward (RH)")
            plt.fill_between(timesteps, agent_3_train_mean - agent_3_train_ci, agent_3_train_mean + agent_3_train_ci, color="green", alpha=0.2)
            plt.xlabel("Training Timesteps")
            plt.ylabel("Mean Episode Reward")
            plt.title(f"RehabBot Mean Reward - Training ({i+1} runs)")
            plt.legend()
            plt.grid(True)
            plt.savefig("training_reward_comparison.png")
            plt.show()
            plt.close()


if __name__ == "__main__":
    main()