#!/usr/bin/env python
""" Basic Stable-Baselines3 hyperparameter optimizer for Gymnasium-compatible environments.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""

__author__ = "Inaki Vazquez"
__email__ = "ivazquez@deusto.es"
__license__ = "GPLv3"

import argparse
import sys
import json
import numpy as np
import importlib

from stable_baselines3 import PPO, DQN, SAC, A2C, DDPG, TD3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import optuna
from optuna.samplers import TPESampler

import gymnasium as gym
from stable_baselines3.common.utils import set_random_seed
from torch import nn
import os

import rehabbot

def optimize_ppo(trial, env, total_timesteps, study_name=None):
	"""Objective function for Optuna to optimize PPO hyperparameters."""
	
	# Define the hyperparameter search space
	learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
	n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096, 8192])
	# Ensure batch_size is a factor of n_steps
	batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
	gamma = trial.suggest_float("gamma", 0.8, 0.9999)
	gae_lambda = trial.suggest_float("gae_lambda", 0.8, 1.0)
	clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
	ent_coef = trial.suggest_float("ent_coef", 1e-8, 0.1)
	vf_coef = trial.suggest_float("vf_coef", 0.1, 1.0)
	max_grad_norm = trial.suggest_float("max_grad_norm", 0.1, 5.0)

	num_layers = trial.suggest_int("num_layers", 1, 4)
	hidden_nodes = trial.suggest_categorical("hidden_nodes_per_layer", [8, 16, 32, 64, 128, 256, 512])
	net_arch = [hidden_nodes for _ in range(num_layers)]
	
	# Create the model with sampled hyperparameters
	model = PPO(
		"MlpPolicy",
		env,
		learning_rate=learning_rate,
		n_steps=n_steps,
		batch_size=batch_size,
		gamma=gamma,
		gae_lambda=gae_lambda,
		clip_range=clip_range,
		ent_coef=ent_coef,
		vf_coef=vf_coef,
		max_grad_norm=max_grad_norm,
		policy_kwargs={"net_arch": net_arch},
		verbose=0,
		tensorboard_log="./optuna_results/logs"
	)
	
	model.learn(total_timesteps=total_timesteps, tb_log_name=study_name, progress_bar=True)
	mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=50)
	return mean_reward

def optimize_sac(trial, env, total_timesteps, study_name=None):
	"""Objective function for Optuna to optimize SAC hyperparameters."""
	
	# Define the hyperparameter search space
	learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
	batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512, 1024])
	tau = trial.suggest_float("tau", 0.001, 0.1)
	gamma = trial.suggest_float("gamma", 0.8, 0.9999)
	train_freq = trial.suggest_int("train_freq", 1, 10)
	gradient_steps = trial.suggest_int("gradient_steps", 1, 10)
	ent_coef = trial.suggest_float("ent_coef", 0.0001, 0.1)
	target_update_interval = trial.suggest_int("target_update_interval", 1, 10)

	# SAC specific: Q-networks and policy architecture
	num_layers = trial.suggest_int("num_layers", 1, 4)
	hidden_nodes = trial.suggest_categorical("hidden_nodes", [64, 128, 256, 512])
	net_arch = [hidden_nodes for _ in range(num_layers)]
	
	# Create the SAC model with sampled hyperparameters
	model = SAC(
		"MlpPolicy",
		env,
		learning_rate=learning_rate,
		batch_size=batch_size,
		tau=tau,
		gamma=gamma,
		train_freq=train_freq,
		gradient_steps=gradient_steps,
		ent_coef=ent_coef,
		target_update_interval=target_update_interval,
		policy_kwargs={"net_arch": net_arch},
		verbose=0,
		tensorboard_log="./optuna_results/logs"
	)
	
	model.learn(total_timesteps=total_timesteps, tb_log_name=study_name, progress_bar=True)
	mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=50)
	return mean_reward

def main():

	parser = argparse.ArgumentParser(description='Search the hyperparameter space for an SB3 algorithm and saves the optuna study.')
	parser.add_argument('-e', '--env', type=str, required=True, help='environment to test (e.g. CartPole-v1)')
	parser.add_argument('-a', '--algo', type=str, required=True,
						help='algorithm (PPO, SAC) to apply from SB3')
	parser.add_argument('-n', '--nsteps', type=int, default=400_000, help='number of steps per optuna trial')
	parser.add_argument('-l', '--ntrials', type=int, default=100, help='number of optuna trials')
	parser.add_argument('--name', type=str, default="study", help='name of this study (for logs and policies)')
	parser.add_argument('-i', '--envpackage', type=str, default=None, help='python package with the environment if not included in Gymnasium')
	parser.add_argument('-m', '--envparams', type=str, default=None, help='path to json file with environment parameters to be passed during creation')
	parser.add_argument('--nenvs', type=int, default=1, help='the number of environments to train in parallel (default=1)')

	args = parser.parse_args()

	str_env = args.env
	str_algo = args.algo
	n_steps = args.nsteps
	n_trials = args.ntrials
	study_name = f"{args.name}_{str_algo}"
	envparams = json.load(open(args.envparams)) if args.envparams else {}
	n_envs = args.nenvs
	if args.envpackage:
		importlib.import_module(args.envpackage)

	seed = 42
	set_random_seed(seed=seed)

	# Create environment
	env = make_vec_env(lambda: gym.make(str_env, render_mode=None, **envparams), n_envs=n_envs)

	os.system("mkdir -p ./optuna_results/")

	# Optuna study configuration
	storage_file = f"sqlite:///optuna_results/optuna.db"
	full_study_dir_path = f"optuna_results/{study_name}"
	tpe_sampler = TPESampler(seed=seed) # For reproducibility
	study = optuna.create_study(sampler=tpe_sampler, direction='maximize', study_name=study_name, storage=storage_file, load_if_exists=True)

	# Start the study
	print(f"Searching for the best {str_algo} hyperparameters in {n_trials} trials...")
	if str_algo == "PPO":
		obj_fun = lambda trial: optimize_ppo(trial, env=env, total_timesteps=n_steps, study_name=study_name)
	elif str_algo == "SAC":
		obj_fun = lambda trial: optimize_sac(trial, env=env, total_timesteps=n_steps, study_name=study_name)

	study.optimize(obj_fun, n_trials=n_trials)

	env.close()

	best_trial = study.best_trial

	# Generate the policy_kwargs key before writing to file
	best_trial_params = json.dumps(best_trial.params, sort_keys=True, indent=4)

	# Save the data in a JSON file
	os.system(f"mkdir -p {full_study_dir_path}")
	best_trial_file = open(f"{full_study_dir_path}/best_trial_{str_algo}.json", "w")
	best_trial_file.write(best_trial_params)
	best_trial_file.close()

if __name__ == "__main__":
	main()