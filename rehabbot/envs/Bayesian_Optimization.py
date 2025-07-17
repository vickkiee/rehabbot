import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
import matplotlib.pyplot as plt
from skopt.plots import plot_convergence, plot_evaluations
from skopt import dump
import numpy as np
import warnings
import torch

import rehabbot


# Filter warnings to reduce output noise
warnings.filterwarnings('ignore')



# Define hyperparameter search space
space = [
    Real(1e-5, 1e-3, name='learning_rate', prior='log-uniform'),
    Real(0.001, 0.1, name='ent_coef', prior='log-uniform'),
    Real(0.001, 0.02, name='tau'),
    Integer(10000, 1000000, name='buffer_size'),
    Categorical([32, 64, 128, 256, 512], name='batch_size'),
    Real(0.9, 0.9999, name='gamma'),
    Categorical(['relu', 'tanh', 'elu'], name='activation_fn'),
]

@use_named_args(space)
def objective(learning_rate, ent_coef, tau, buffer_size, batch_size, gamma, activation_fn):
    """
    Objective function to minimize (negative of average reward)
    """
    try:
        # Create environment
        env = make_vec_env('rehabbot/rehab-reach-l-v0', n_envs=1)

        
        # Create model with current hyperparameters
        model = SAC(
            'MlpPolicy',
            env,
            learning_rate=learning_rate,
            ent_coef=ent_coef,
            tau=tau,
            buffer_size=buffer_size,
            batch_size=batch_size,
            gamma=gamma,
            policy_kwargs=dict(activation_fn=dict(relu=torch.nn.ReLU, 
                                                 tanh=torch.nn.Tanh,
                                                 elu=torch.nn.ELU)[activation_fn]),
            verbose=0,
            seed=42
        )
        
        # Train the model
        model.learn(total_timesteps=300000, progress_bar=True)
        
        # Evaluate performance
        mean_reward, _ = evaluate(model, env)
        env.close()
        
        # Print progress
        print(f"Params: LR={learning_rate:.5f}, ent={ent_coef}, "
              f"tau={tau:.4f}, buf={buffer_size}, batch={batch_size}, tar_ent={target_entropy}, "
              f"γ={gamma:.3f}, act={activation_fn} → Reward: {mean_reward:.2f}")
        
        #print(np.prod(env.action_space.shape))
        
        # Return negative reward (since we minimize)
        return -mean_reward
    
    except Exception as e:
        # Return large penalty for invalid configurations
        print(f"Error with params: {e}")
        return 1000

def evaluate(model, env, n_eval_episodes=10):
    """
    Evaluate model performance
    """
    episode_rewards = []
    for _ in range(n_eval_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        episode_rewards.append(total_reward)
    return np.mean(episode_rewards), np.std(episode_rewards)

# Run Bayesian optimization
res = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=30,             # Number of evaluations
        n_random_starts=10,     # Random initial points
        random_state=42,
        verbose=True,
        acq_func='EI'           # Expected Improvement
    )


# Create and save plots
conv_ax = plot_convergence(res)  # Returns Axes object
conv_fig = conv_ax.get_figure()  # Get the figure from the axes
conv_fig.savefig("convergence_plot.png")
plt.close(conv_fig)

eval_ax = plot_evaluations(res)  # Returns Figure and Axes
eval_fig = eval_ax.get_figure()
eval_fig.savefig("evaluations_plot.png")
plt.close(eval_fig)

# Print best results
print("\n=== Optimization Results ===")
print(f"Best reward: {-res.fun:.2f}")
print("Best parameters:")
print(f"Learning Rate: {res.x[0]:.6f}")
print(f"Entropy Coef: {res.x[1]:.4f}")
print(f"Tau: {res.x[2]:.4f}")
print(f"Buffer Size: {res.x[3]}")
print(f"Batch Size: {res.x[4]}")
print(f"Gamma: {res.x[5]:.4f}")
print(f"Activation: {res.x[6]}")


