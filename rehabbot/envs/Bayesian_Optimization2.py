import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
import matplotlib.pyplot as plt
from skopt.plots import plot_convergence, plot_evaluations
from skopt import dump, load
import numpy as np
import warnings
import torch
import os
import hashlib
import json
import torch.nn as nn
import rehabbot

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Filter warnings to reduce output noise
warnings.filterwarnings('ignore')

# Define hyperparameter search space
space = [
    Real(1e-5, 1e-3, name='learning_rate', prior='log-uniform'),
    Real(0.001, 0.1, name='ent_coef', prior='log-uniform'),
    Real(0.001, 0.02, name='tau'),
    Integer(900000, 1000000, name='buffer_size'),
    Categorical([64, 128, 256, 512], name='batch_size'),
    Real(0.9, 0.9999, name='gamma'),
    Categorical(['relu', 'tanh', 'elu'], name='activation_fn'),
]

# Helper function to convert numpy types to native Python types
def convert_to_native_types(params):
    """Convert numpy types to native Python types for JSON serialization"""
    converted = {}
    for key, value in params.items():
        if isinstance(value, np.generic):
            # Convert numpy scalars to Python types
            converted[key] = value.item()
        elif isinstance(value, np.ndarray):
            # Convert arrays to lists
            converted[key] = value.tolist()
        else:
            converted[key] = value
    return converted

# Unique identifier for each hyperparameter config
def get_config_id(params):
    """Create unique hash for hyperparameter set"""
    # Ensure we're using native types
    native_params = convert_to_native_types(params)
    return hashlib.sha256(
        json.dumps(native_params, sort_keys=True).encode()
    ).hexdigest()[:12]

# Custom save function with replay buffer
def save_sac_with_buffer(model, path, params):
    """Save model with hyperparams and replay buffer"""
    # Create save directory
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save standard model components
    model.save(path)
    
    # Save replay buffer separately
    buffer_path = path.replace(".zip", "_replay_buffer.pkl")
    model.save_replay_buffer(buffer_path)
    
    # Save hyperparameters (convert to native types first)
    param_path = path.replace(".zip", "_params.json")
    with open(param_path, "w") as f:
        json.dump(convert_to_native_types(params), f, indent=2)

# Custom load function with replay buffer
def load_sac_with_buffer(path, env):
    """Load model with replay buffer and hyperparams"""
    # Load base model
    model = SAC.load(path, env=env)
    
    # Try to load replay buffer
    buffer_path = path.replace(".zip", "_replay_buffer.pkl")
    if os.path.exists(buffer_path):
        model.load_replay_buffer(buffer_path)
    
    # Load hyperparameters
    param_path = path.replace(".zip", "_params.json")
    params = {}
    if os.path.exists(param_path):
        with open(param_path, "r") as f:
            params = json.load(f)
    
    return model, params

@use_named_args(space)
def objective(**params):
    """
    Objective function to minimize (negative of average reward)
    """
    # Convert all parameters to native types immediately
    native_params = convert_to_native_types(params)
    config_id = get_config_id(native_params)
    save_path = f"sac_models/{config_id}/model.zip"
    
    env = None
    try:
        # Create environment
        env = make_vec_env('rehabbot/rehab-reach-l-v0', n_envs=1, seed=SEED)
        
        if os.path.exists(save_path):
            model, loaded_params = load_sac_with_buffer(save_path, env)
            print(f"Resumed training for config {config_id} at {model.num_timesteps} steps")
        else:
            # Map activation function string to actual PyTorch module
            activation_fn = {
                'relu': nn.ReLU,
                'tanh': nn.Tanh,
                'elu': nn.ELU
            }[params['activation_fn']]
            
            policy_kwargs = dict(activation_fn=activation_fn)
            
            # Create model with current hyperparameters
            model = SAC(
                'MlpPolicy',
                env,
                learning_rate=params['learning_rate'],
                ent_coef=params['ent_coef'],
                tau=params['tau'],
                buffer_size=params['buffer_size'],
                batch_size=params['batch_size'],
                gamma=params['gamma'],
                policy_kwargs=policy_kwargs,
                verbose=1,
                seed=SEED
            )
            print(f"Started new training for config {config_id}")
        
        # Train the model
        model.learn(total_timesteps=50000, progress_bar=True, reset_num_timesteps=False)
        
        # Save model with buffer
        save_sac_with_buffer(model, save_path, params)
        
        # Evaluate performance
        mean_reward = evaluate(model, env)
        env.close()
        
        # Print progress
        print(f"Params: LR={params['learning_rate']:.5f}, ent={params['ent_coef']}, "
              f"tau={params['tau']:.4f}, buf={params['buffer_size']}, batch={params['batch_size']}, "
              f"γ={params['gamma']:.3f}, act={params['activation_fn']} → Reward: {mean_reward:.2f}")
        
        # Return negative reward (since we minimize)
        return -mean_reward
    
    except Exception as e:
        # Close environment if created
        if env is not None:
            env.close()
        # Return large penalty for invalid configurations
        # Use native_params for printing to ensure serialization
        print(f"Error with params {native_params}: {str(e)}")
        return 1000

def evaluate(model, env, n_eval_episodes=10):
    """
    Evaluate model performance
    """
    episode_rewards = []
    for _ in range(n_eval_episodes):
        obs = env.reset()
        done = [False]  # Initialize for vectorized env
        total_reward = 0
        
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward[0]  # Handle vectorized reward
            
        episode_rewards.append(total_reward)
    return np.mean(episode_rewards)

# Initialize with no previous results
res = None
checkpoint_file = "bayes3_opt_checkpoint.pkl"

# Check for existing optimization checkpoint
if os.path.exists(checkpoint_file):
    res = load(checkpoint_file)
    print(f"Resuming optimization with {len(res.x_iters)} existing evaluations")

# Run Bayesian optimization
res = gp_minimize(
        func=objective,
        dimensions=space,
        x0=res.x_iters if res else None,     # Previous evaluation points
        y0=res.func_vals if res else None,   # Previous results
        n_calls=30,             # Number of evaluations
        n_random_starts=10,     # Random initial points
        random_state=SEED,
        verbose=True,
        acq_func='EI'           # Expected Improvement
    )

# Save optimization state
dump(res, checkpoint_file)

# Create and save plots
conv_ax = plot_convergence(res)
conv_fig = conv_ax.get_figure()
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