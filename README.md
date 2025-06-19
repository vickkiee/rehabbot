# RehabBot: Mujoco-Based Rehabilitation Environment for Upper-Limb Physiotherapy-Based Reaching Tasks

![Rehabilitation Simulation Demo](docs/demo.gif)  
*Example of DRL-controlled rehabilitation exercise with variable limb mass*

This repository contains a MuJoCo-based simulation environment and Deep Reinforcement Learning (DRL) implementation for adaptive rehabilitation robotics. The system combines a UR5e robotic arm with a humanoid model to perform trajectory tracking exercises for shoulder and elbow joints, adapting dynamically to different upper-arm weights.

## Key Features
- 🦾 Biomechanical Integration: UR5e robot coupled with humanoid upper-limb dynamics
- 🧠 Adaptive DRL Policy: Single policy that adapts to variable arm masses (0.5× to 1.5× nominal)
- 📊 Algorithm Benchmarking: Comparison of PPO, SAC, and TD3 with Bayesian-tuned hyperparameters
- 🎯 Clinical Trajectories: Sagittal/frontal plane reaching and spiral tracing exercises
- 📈 Performance Metrics: Tracking error, torque smoothness, and adaptation gap analysis

## Repository Structure
.
├── envs/                         # MuJoCo simulation environments
│   ├── UR5e_humanoid.xml         # Main robot-humanoid integration
│   ├── humanoid_arm.xml          # Humanoid upper-limb submodel
│   └── ur5e_with_gripper.xml     # UR5e robotic arm model
│
├── src/                          # Training and evaluation code
│   ├── train.py                  # Main training script
│   ├── environment.py            # Custom Gym environment
│   ├── policies/                 # DRL policy architectures
│   │   ├── mass_conditioned.py   # Mass-adaptive policy
│   │   └── attention_policy.py   # Transformer-enhanced policy
│   │
│   ├── configs/                  # Hyperparameter configurations
│   │   ├── ppo_params.yaml       # PPO configuration
│   │   ├── sac_params.yaml       # SAC configuration
│   │   └── td3_params.yaml       # TD3 configuration
│   │
│   └── utils/                    # Helper utilities
│       ├── trajectory_generators.py # Rehabilitation trajectories
│       └── visualization.py      # Result plotting tools
│
├── trained_models/               # Pre-trained policy weights
│   ├── sac_adaptive_policy.zip   # Best-performing SAC model
│   └── ppo_baseline.zip          # Baseline PPO model
│
├── docs/                         # Documentation and media
│   ├── architecture.pdf          # System architecture diagram
│   └── performance_results.pdf   # Benchmark results
│
├── requirements.txt              # Python dependencies
└── Dockerfile                    # Container configuration
## Installation

### Prerequisites
- Python 3.8+
- [MuJoCo 2.3.0+](https://mujoco.org/)
- NVIDIA GPU (recommended)

### Setup
```bash
# Clone repository
git clone https://github.com/vickkiee/rehabbot.git 
cd rehabbot

# Create virtual environment
python -m venv rehab_env
source rehab_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start


### Training the Adaptive Policy

```python
from src import train

# Train SAC policy with mass adaptation
train.run_experiment(
    algorithm="SAC",
    env_config="shoulder_elbow_rehab",
    policy="MassConditioned",
    params_path="src/configs/sac_params.yaml",
    total_timesteps=1_000_000
)
```

### Evaluating a Trained Policy

```python
from src.environment import RehabilitationEnv
from stable_baselines3 import SAC

# Load environment with increased arm mass
env = RehabilitationEnv(
    humanoid_model="envs/humanoid_arm.xml",
    robot_model="envs/ur5e_with_gripper.xml",
    arm_mass=1.25  # 25% above nominal mass
)

# Load trained policy
model = SAC.load("trained_models/sac_adaptive_policy.zip")

# Run evaluation
mean_reward, _ = evaluate_policy(
    model, 
    env, 
    n_eval_episodes=10,
    deterministic=True
)
print(f"Mean reward: {mean_reward:.2f}")
```

### Visualizing Results
```python
from src.utils.visualization import plot_tracking_performance


# Compare algorithm performance
plot_tracking_performance(
    log_paths=[
        "logs/ppo_nominal",
        "logs/sac_adaptive",
        "logs/td3_baseline"
    ],
    metrics=["tracking_error", "torque_smoothness"],
    save_path="results/performance_comparison.png"
)
```

## Key Configuration Options
| Parameter | Description | Default |
|-----------|-------------|---------|
| `arm_mass_range` | Arm mass variation range | [0.5, 1.5] |
| `trajectory_type` | Exercise pattern (reaching, spiral) | "sagittal_reaching" |
| `safety_penalty` | Joint limit violation penalty | 10.0 |
| `max_torque` | Maximum joint torque (Nm) | 150 |
| `control_frequency` | Control update rate (Hz) | 50 |

## Benchmark Results
| Algorithm | Tracking Error (mm) | Adaptation Gap | Torque Variance |
|-----------|----------------------|----------------|-----------------|
| SAC | 3.2 ± 0.4 | 14% | 0.08 |
| TD3 | 4.1 ± 0.7 | 38% | 0.12 |
| PPO | 5.3 ± 1.2 | 61% | 0.15 |

*Performance at ±50% mass variation (lower values better)*

## Citing This Work
```bibtex
@misc{adaptive_rehab_2025,
  title = {Adaptive Rehabilitation Robotics: DRL Framework for Variable-Load Trajectory Tracking},
  author = {Victoria Oguntosin},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/vickkiee/rehabbot}}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For questions or collaborations, contact:  
[Victoria Oguntosin] - [v.oguntosin@deusto.es]  
Project Link: [https://github.com/vickkiee/rehabbot](https://github.com/vickkiee/rehabbot)