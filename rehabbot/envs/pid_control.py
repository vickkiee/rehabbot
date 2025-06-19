from __future__ import annotations
from typing import Any

import time
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.core import ObsType, ActType, SupportsFloat, RenderFrame
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.wrappers import TimeLimit
import mujoco
import numpy as np
import math
import mujoco.viewer
from robot_descriptions import ur5e_mj_description
import os

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy

import logging
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo


class RehabEnv(gym.Env):

    metadata = {'render_modes': ['human'], 'render_fps': 30}
    
    def __init__(self, render_mode=None):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        #self.model = mujoco.MjModel.from_xml_path(current_dir + "/xml_files/humanoid_UR5E.xml")
        self.model = mujoco.MjModel.from_xml_path(current_dir + "/xml_files/human_UR5E2.xml")
        self.data = mujoco.MjData(self.model)
        
        # Control parameters
        self.kp = 0.5  # Proportional gain
        self.ki = 0.01
        self.kd = 0.1
        self.integral = 0
        self.prev_error = 0
        self.tolerance = 0.09  # Radian tolerance for position reached
        
        # Create a persistent data copy for FK calculations
        self.fk_data = mujoco.MjData(self.model)
        
        # Target trajectory (shape: [10, 6] for 10 targets x 6 joints)
        self.target_pos = [[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0]  ]
        self.current_target_idx = 0
        self.total_steps = 0
        self.step_counter = 0

        self.render_mode = render_mode
        if self.render_mode == "human":
            # Create a viewer to visualize the simulation
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        # Target pos, current pos of ee, joint positions
        self.observation_space = gym.spaces.Box(low=np.array([-2*math.pi]*6 + [-1]*3 + [-2*math.pi]*6, dtype=np.float32), high=np.array([+2*math.pi]*6 + [+1]*3 + [+2*math.pi]*6, dtype=np.float32), shape=(15,))
        action_max = 2*math.pi / 1000
        self.action_space = gym.spaces.Box(low=np.array([-action_max]*6, dtype=np.float32), high=np.array([+action_max]*6, dtype=np.float32), shape=(6,))


    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
        np.random.seed(seed)
        mujoco.mj_resetData(self.model, self.data)  # Fully resets simulation state
        
        self.target_pos = np.array([ [0.6786, 0.5246, -0.154, 0, 0, 0],
                        [0.6032, 0.2952, -0.308, 0, 0, 0],
                        [0.5278, 0.0658, -0.462, 0, 0, 0],
                        [0.4524, -0.1636, -0.616, 0, 0, 0],
        ])
        
        self.current_target_idx = 0


        if self.render_mode == "human":
            link_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")
            self.draw_sphere(self.data.xpos[link_id], 0.05)
        return self.get_observation(), {}
        
    def wait_until_stable(self, sim_steps=500):
        joint_pos = self.get_observation()[3:6]
        for _ in range(sim_steps):
            mujoco.mj_step(self.model, self.data)
            if self.render_mode == "human":
                self.viewer.sync()
            new_joint_pos = self.get_observation()[3:6]
            if np.sum(np.abs(np.array(joint_pos)-np.array(new_joint_pos))) < 5e-3: # Threshold based on experience
                return True
            joint_pos = new_joint_pos
        print("Warning: The robot configuration did not stabilize")
        return False

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        
        assert self.action_space.contains(action), f"Invalid Action: {action}"
        self.data.ctrl[21:24] += action[:3]
        self.data.ctrl[21] = np.clip(self.data.ctrl[21], 0, math.pi/4)
        self.data.ctrl[22] = np.clip(self.data.ctrl[22], -math.pi/2, math.pi/4)
        self.data.ctrl[23] = np.clip(self.data.ctrl[23], -math.pi/2, 0)
        
        
        current_target_idx = 0
        target_reached = False
        
        # Get current target
        target = self.target_pos[self.current_target_idx]
        dt = 0.001
        
        # Calculate control signal (simple P-control)
        error = target[0:3] - self.data.qpos[28:31] 
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.data.ctrl[21:24] = self.kp * error[0:3] + self.ki * self.integral[0:3] + self.kd * derivative[0:3]
        self.wait_until_stable()
        
        obs = self.get_observation()
        joint_pos = obs[9:15]
 
 
        # 2. Extended Physics Stepping
        target_reached = False
        for _ in range(50):  # Increased physics substeps
            mujoco.mj_step(self.model, self.data)
            
            # Update current state
            targeted_joints = self.target_pos[self.current_target_idx]         
            error = np.linalg.norm(joint_pos[0:3] - targeted_joints[0:3])
            
            # Early exit if close to target
            if error < 0.09:  
                target_reached = True
                break
        
        # 3. Target Progression Logic
        self.step_counter += 1
        
        # Progressive timeout: longer allowance for later targets
        timeout = 500 + 100 * self.current_target_idx  # 500-1500 steps
        truncated = self.step_counter > timeout
        
        # 4. Enhanced Reward Structure
        targeted_joints = self.target_pos[self.current_target_idx] 
        target_error = np.linalg.norm(joint_pos[0:3] - targeted_joints[0:3])

        reward = -target_error
        
        # 5. Target Advancement Conditions
        if target_reached or truncated:
            if target_reached:
                reward += 100
                self.current_target_idx += 1
                print(f"Target {self.current_target_idx-1} reached in {self.step_counter} steps")
                
            if self.current_target_idx >= len(self.target_pos) or truncated:
                terminated = True
                if truncated:
                    print(f"Episode truncated at target {self.current_target_idx}")
                else:
                    print("All targets completed!")
            else:
                # Reset step counter for new target
                self.step_counter = 0
                
            # Force reset if stuck
            if truncated and self.current_target_idx > 0:
                self.current_target_idx -= 1  # Retry previous target
                
        else:
            terminated = False        
   
        
        
        info = {}
        return obs, reward, terminated, truncated, info

    def get_target_ee_pose(self):
        # Reset FK data to current model defaults
        mujoco.mj_resetData(self.model, self.fk_data)
        self.fk_data.qpos[28:34] = self.target_pos[self.current_target_idx] 
        # Compute forward kinematics on the copy
        mujoco.mj_forward(self.model, self.fk_data)
        link_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")
        
        return self.fk_data.xpos[link_id]
    
    def render(self) -> RenderFrame | list[RenderFrame] | None:
        self.viewer.sync()

    def close(self):
        if self.render_mode == 'human':
            self.viewer.close()

    def get_observation(self):
        ee_pos, ee_quat = self.get_ee_pose()
        joint_pos = self.data.qpos[28:34]
        return np.concatenate((self.target_pos[self.current_target_idx], ee_pos, joint_pos), dtype=np.float32)

    def get_ee_pose(self):
        link_name = "wrist_3_link"
        link_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, link_name)
        return self.data.xpos[link_id], self.data.xquat[link_id]
    
    def draw_sphere(self, center, radius, color=(1, 0, 0, 1)):
        """Draws a debug sphere at a given 3D position in MuJoCo."""
        self.viewer.user_scn.ngeom = 0
        mujoco.mjv_initGeom(
                self.viewer.user_scn.geoms[0],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[radius, 0, 0],
                pos=center,
                mat=np.eye(3).flatten(),
                rgba=color
            )
        self.viewer.user_scn.ngeom = 1
        


        
        

import time
import rehabbot
if __name__ == "__main__":
    env = gym.make("rehabbot/pid-control", render_mode="human")
    env = TimeLimit(env, max_episode_steps=100)
    
    for _ in range(100):
        print("Resetting")
        obs = env.reset()
        for _ in range(1000):
            action = [0]*6
            action =  env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            #print("Current joint angles", obs[9:15]) #Print joint angles
            #print("Current end effector pose", obs[6:9]) #Print end effector pose
            env.render()
            if truncated:
                break
    env.close()
    
    

    