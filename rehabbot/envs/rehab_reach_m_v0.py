from __future__ import annotations
from typing import Any

import time
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.core import ObsType, ActType, SupportsFloat, RenderFrame
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.wrappers import TimeLimit
import mujoco
import numpy as np
import math
import mujoco.viewer
from robot_descriptions import ur5e_mj_description
import os
import random

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy

import logging
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo


class RehabEnv(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array", ],
    }

    def __init__(self, **kwargs):

        default_camera_config = {
            "distance": 2.5,
            "elevation": -30.0,
            "azimuth": 90.0,
            "lookat": [0.0, 0.0, 0.6],
        }

        screen_width = screen_height = 800

        MujocoEnv.__init__(
            self,
            model_path=os.path.dirname(os.path.abspath(__file__)) + "/xml_files/humanoid_UR5E2.xml",
            frame_skip=5,
            observation_space=None,
            default_camera_config=default_camera_config,
            width=screen_width,
            height=screen_height,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }
        

        # Create a data structure to hold the simulation state
        self.data = mujoco.MjData(self.model)
        
        # Create a persistent data copy for FK calculations
        self.fk_data = mujoco.MjData(self.model)

        
        if self.render_mode == "human":
            # Create a viewer to visualize the simulation
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        # Target pos, current pos of ee, joint positions
        self.observation_space = gym.spaces.Box(low=np.array([-2*math.pi]*6 + [-1]*3 + [-2*math.pi]*6 + [-1]*3 + [-2*math.pi]*3 + [0]*1 + [0]*1, dtype=np.float32), high=np.array([+2*math.pi]*6 + [+1]*3 + [+2*math.pi]*6 + [+1]*3 + [+2*math.pi]*3 + [4]*1 + [1]*1, dtype=np.float32), shape=(23,))
        action_max = 2*math.pi / 800
        self.action_space = gym.spaces.Box(low=np.array([-action_max]*6, dtype=np.float32), high=np.array([+action_max]*6, dtype=np.float32), shape=(6,))
        
        # Target trajectory (shape: [10, 6] for 10 targets x 6 joints)
        self.target_pos = [[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0]  ]
        self.current_target_idx = 0
        self.current_trajectory_idx = 0
        
        self.target_mass = [[0, 0, 0],[0, 0, 0],[0, 0, 0],[0, 0, 0]  ]
        self.current_mass_idx = 0
        
        start = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")
        self.prev_pos = self.data.xpos[start]
        
        #self.human_model = PPO.load("")


    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
        np.random.seed(seed)
        mujoco.mj_resetData(self.model, self.data)  # Fully resets simulation state

        target_pos2 = np.array([ [0.1636, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0], 
                        [-0.09, 0, 0.0754, 0, 0, 0],
                        [-0.18, 0, 0.0754, 0, 0, 0],
                        [-0.27, 0, 0.1508, 0, 0, 0],
                        [-0.36, 0, 0.1508, 0, 0, 0],
                        [-0.54, 0, 0.2262, 0, 0, 0],
                        [-0.63, 0, 0.2262, 0, 0, 0],
                        [-0.70, 0, 0.3016, 0, 0, 0],
                        [-0.785, 0, 0.3016, 0, 0, 0],
        ])  
        
        target_pos1 = np.array([ [0, -0.16, 0.0754, 0, 0, 0],
                        [0, 0, 0.0754, 0, 0, 0],
                        [0, 0.1, 0.1508, 0, 0, 0],
                        [0, 0.2, 0.1508, 0, 0, 0],
                        [0, 0.3, 0.2262, 0, 0, 0],
                        [0, 0.4, 0.2262, 0, 0, 0],
                        [0, 0.5, 0.3016, 0, 0, 0],
                        [0, 0.6, 0.3016, 0, 0, 0],
                        [0, 0.7, 0.377, 0, 0, 0],
                        [0, 0.8, 0.377, 0, 0, 0],
        ])
        
        target_poses = np.empty(2, dtype=object)
        target_poses[0] = target_pos1
        target_poses[1] = target_pos2
        
        self.current_trajectory_idx = random.randint(0,1)
        self.target_pos = target_poses[self.current_trajectory_idx]

        self.current_target_idx = 0

        start = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")
        self.prev_pos = self.data.xpos[start]
        
        #self.target_mass = [[0.017, 0.009, 0.002],[0.17, 0.09, 0.02],[0.34, 0.18, 0.04],[0.68, 0.36, 0.08] ]  
        self.target_mass = [[0.017, 0.009, 0.002],[0.17, 0.09, 0.02],[0.34, 0.18, 0.04],[0.68, 0.36, 0.08],[0.85, 0.45, 0.10] ]  
        #self.target_mass = [[0.17, 0.09, 0.02],[0.34, 0.18, 0.04],[0.68, 0.36, 0.08], [1.02, 0.54, 0.2], [1.20, 0.65, 0.3], ] 
        
        self.current_mass_idx = random.randint(0,4)
        
        self.model.body_mass[14:17] = self.target_mass[self.current_mass_idx]
        print(self.model.body_mass[14:17], " with current_trajectory_idx = ", self.current_trajectory_idx)
        
        if self.render_mode == "human":
            link_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")
            self.draw_sphere(self.data.xpos[link_id], 0.05, (0, 1, 0, 1))
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
        
        #if np.all(self.data.qfrc_actuator[22:25] >= - 1) and np.all(self.data.qfrc_actuator[22:25] <= 1):   # if humanoid isn´t strong enough, the UR5E should move to the target
        
        
        
        self.data.ctrl[21:24] += action[:3]
        self.data.ctrl[21] = np.clip(self.data.ctrl[21], -math.pi/4, math.pi/12)
        self.data.ctrl[22] = np.clip(self.data.ctrl[22], -math.pi/12, math.pi/3.5)
        self.data.ctrl[23] = np.clip(self.data.ctrl[23], 0, math.pi/8)
        self.wait_until_stable()
         
        target = self.target_pos[self.current_target_idx]
        obs = self.get_observation() 
        joint_pos = obs[9:15]
        distance = np.linalg.norm(target[:3] - joint_pos[:3])
        position_error = np.abs(target - joint_pos)
        reward = -distance
        terminated = False
        if np.all(position_error < 9e-2):
            if self.current_target_idx < len(self.target_pos) - 1: 
                self.current_target_idx += 1
                reward += 10  # Bonus for reaching target
                print(f"Reached target {self.current_target_idx-1}, moving to {self.current_target_idx}")  
            else:
                terminated = True  # All targets reached
                reward += 100
                print("All targets reached!")       
        if self.render_mode == "human":
            link_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")
            self.draw_sphere(self.get_target_ee_pose(), 0.03, (0, 1, 0, 1))     
            #self.add_visual_capsule(self.prev_pos, self.data.xpos[link_id], 0.009)     
            #self.prev_pos = self.data.xpos[link_id].copy()
        
        
        #human_actions = self.human_model.predict(self.get_observation(), deterministic=True)
        #self.data.ctrl[15:18] += human_actions
        
        
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info
        
        

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        self.viewer.sync()

    def close(self):
        if self.render_mode == 'human':
            self.viewer.close()
        
    def get_observation(self):
        ee_pos = self.get_target_ee_pose()
        joint_pos = self.data.qpos[28:34]     
        humanoid_joint_pos = self.data.qpos[25:28]
        humanoid_force_applied = self.data.qfrc_passive[25:28] + self.data.qfrc_actuator[25:28] + self.data.qfrc_applied[25:28]
        my_current_pos = self.get_current_ee_pose()
        target = self.target_pos[self.current_target_idx]
        traj_level = [self.current_trajectory_idx]
        level = [self.current_mass_idx]
        return np.concatenate((target, ee_pos, joint_pos, my_current_pos, humanoid_joint_pos, level, traj_level), dtype=np.float32)

    def get_current_ee_pose(self):
        link_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")
        return self.data.xpos[link_id]
        

    def get_target_ee_pose(self):
        mujoco.mj_resetData(self.model, self.fk_data)
        self.fk_data.qpos[28:34] = self.target_pos[self.current_target_idx]
        mujoco.mj_forward(self.model, self.fk_data)
        link_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")
        return self.fk_data.xpos[link_id]
        
    
    def draw_sphere(self, center, radius, color=(1, 0, 0, 1)):
        """Draws a debug sphere at a given 3D position in MuJoCo."""
        self.viewer.user_scn.ngeom = 0
        mujoco.mjv_initGeom(self.viewer.user_scn.geoms[0], type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[radius, 0, 0], pos=center, mat=np.eye(3).flatten(), rgba=color )
        self.viewer.user_scn.ngeom = 1
        
        
    def add_visual_capsule(self, point1, point2, radius, color=(0.5, 0, 0.5, 0.8)):
        current_count = self.viewer.user_scn.ngeom
        max_geoms = self.viewer.user_scn.maxgeom
        if current_count >= max_geoms:
            return
        self.viewer.user_scn.ngeom += 1  # increment ngeom
        # initialise a new capsule, add it to the scene using mjv_connector
        mujoco.mjv_initGeom(self.viewer.user_scn.geoms[current_count-1], mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.zeros(9), rgba=color)
        mujoco.mjv_connector(self.viewer.user_scn.geoms[current_count-1], mujoco.mjtGeom.mjGEOM_LINE, radius, point1, point2)


        
        

import time
import rehabbot
if __name__ == "__main__":
    env = gym.make("rehabbot/rehab-reach-m-v0", render_mode="human")
    
    env = TimeLimit(env, max_episode_steps=1000)
    
    q_current_0 = []
    q_current_1 = []
    q_current_2 = []
    q_target_0 = []
    q_target_1 = []
    q_target_2 = []
    h_current_0 = []
    h_current_1 = []
    h_current_2 = []

    posx = []
    posy = []
    posz = []
    posx1 = []
    posy2 = []
    posz3 = []
    rewa = []
    dist = []
    


    
    
    
    
    
    
    
    
    
    

    train = False       #False to visualize
    #train = True        #True to train/continue training. 

    if train:
        #model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./UR5e_tensorboard/") #This line IS TO TRAIN FROM SCRATCH - comment to retrain and uncomment below
        model = PPO.load("vm_DRL_model_PPO", env=env, print_system_info=True, tensorboard_log="./UR5e_tensorboard/") #This line IS TO RETRAIN - comment to train from scratch and uncomment above
        model.learn(total_timesteps=int(500000), progress_bar=True, reset_num_timesteps=False)
        model.save("vm_DRL_model_PPO")
    else:
        env = gym.make("rehabbot/rehab-reach-m-v0", render_mode="human")
        model = SAC.load("SAC-1M", env=env, print_system_info=True)
        #model = PPO.load("vm_DRL_model_PPO", env=env, print_system_info=True)

        #mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=5)
    

        # Enjoy trained agent
        vec_env = model.get_env()
        obs = vec_env.reset()
         
        episodes = 10

        for ep in range(episodes):
            #obs = vec_env.reset()
            done = False
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, rewards, done, info = vec_env.step(action)
                vec_env.render("human")
                #print(np.around(obs[0, 9:12],3))
                np.set_printoptions(precision=4, suppress=True)
                #print("target_ang: ", obs[0, 0:3]*360/(2*math.pi), "current_ang: ", obs[0, 9:12]*360/(2*math.pi), "pose: ", obs[0, 6:9], "target pose: ", obs[0, 15:18])
                #print("target_ang: ", obs[0, 0:3], "current_ang: ", obs[0, 9:12], "target pose: ", obs[0, 6:9], "pose: ", obs[0, 15:18])
                #print("Reward: ", rewards, "Done: ", done)
                #print("target_ang: ", obs[0, 0:3], "current_ang: ", obs[0, 9:12], "humanoid_current_ang: ", obs[0, 18:21])
                #print(obs[0, 9:15])
                
                q_target_0.append(obs[0, 0])
                q_target_1.append(obs[0, 1])
                q_target_2.append(obs[0, 2])
                q_current_0.append(obs[0, 9])
                q_current_1.append(obs[0, 10])
                q_current_2.append(obs[0, 11])
                h_current_0.append(obs[0, 18])
                h_current_1.append(obs[0, 19])
                h_current_2.append(obs[0, 20])


                posx.append(obs[0, 15])
                posy.append(obs[0, 16])
                posz.append(obs[0, 17])
                posx1.append(obs[0, 6])
                posy2.append(obs[0, 7])
                posz3.append(obs[0, 8])
                rewa.append(rewards[0])
                #dist.append((-3.0*rewards[0]) ** (1./3))
                dist.append(-rewards[0])
                
                #logging.info(f"episode-{ep}", info["episode"])
                
                

        # plot
        _, ax = plt.subplots(3, 2, sharex=True, figsize=(9, 8))

        lines = ax[0,0].plot(q_target_0,'r')
        ax[0,0].plot(q_current_0,'b')
        ax[0,0].set_title('Shoulder Joint 0')
        ax[0,0].set_ylabel('Angle')
        ax[0,0].legend(list(iter(lines)), ('target', 'current'));

        ax[1,0].plot(q_target_1,'r')
        ax[1,0].plot(q_current_1,'b')
        ax[1,0].set_title('Shoulder Joint 1')
        ax[1,0].set_ylabel('Angle')
        ax[1,0].legend(list(iter(lines)), ('target', 'current'));
        
        ax[2,0].plot(q_target_2,'r')
        ax[2,0].plot(q_current_2,'b')
        ax[2,0].set_title('Elbow Joint')
        ax[2,0].set_ylabel('Angle')
        ax[2,0].legend(list(iter(lines)), ('target', 'current'));

        ax[0,1].plot(np.subtract(q_current_0,q_target_0),'g')
        ax[0,1].set_title('Shoulder Error Joint 0')
        ax[0,1].set_ylabel('Angle')

        ax[1,1].plot(np.subtract(q_current_1,q_target_1),'g')
        ax[1,1].set_title('Shoulder Error Joint 1')
        ax[1,1].set_ylabel('Angle')
        
        ax[2,1].plot(np.subtract(q_current_2,q_target_2),'g')
        ax[2,1].set_title('Elbow Error Joint')
        ax[2,1].set_ylabel('Angle')
        
        
        _, av = plt.subplots(1, 3, sharex=True, figsize=(9, 3))
        av[0].plot(h_current_0,'b')
        av[0].set_title('Human Shoulder1 Angle')
        av[0].set_ylabel('Angle')
        
        av[1].plot(h_current_1,'b')
        av[1].set_title('Human Shoulder2 Angle')
        av[1].set_ylabel('Angle')
        
        av[2].plot(h_current_2,'b')
        av[2].set_title('Human Elbow1 Angle')
        av[2].set_ylabel('Angle')      
        
        _, ay = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
        ay[0].scatter(posx, posy, posz)
        ay[0].set_title('Current Trajectory') 
        ay[1].scatter(posx1, posy2, posz3)
        ay[1].set_title('Target Trajectory')
        
        _, az = plt.subplots(1, 2, sharex=True, figsize=(6, 4))
        az[0].plot(rewa,'r')
        az[0].set_title('Reward')
        az[0].set_ylabel('Reward')
        az[1].plot(dist,'b')
        az[1].set_title('Distance')
        az[1].set_ylabel('Distance')
        
        
        
        _, aw = plt.subplots(3, 2, sharex=True, figsize=(10, 9))

        lines = aw[0,0].plot(posx1,'r')
        aw[0,0].plot(posx,'b')
        aw[0,0].set_title('Pose X')
        aw[0,0].set_ylabel('(m)')
        aw[0,0].legend(list(iter(lines)), ('target', 'current'));

        aw[1,0].plot(posy2,'r')
        aw[1,0].plot(posy,'b')
        aw[1,0].set_title('Pose Y')
        aw[1,0].set_ylabel('(m)')
        aw[1,0].legend(list(iter(lines)), ('target', 'current'));
        
        aw[2,0].plot(posz3,'r')
        aw[2,0].plot(posz,'b')
        aw[2,0].set_title('Pose Z')
        aw[2,0].set_ylabel('(m)')
        aw[2,0].legend(list(iter(lines)), ('target', 'current'));

        aw[0,1].plot(np.subtract(posx,posx1),'g')
        aw[0,1].set_title('Pose X Error')
        aw[0,1].set_ylabel('(m)')

        aw[1,1].plot(np.subtract(posy,posy2),'g')
        aw[1,1].set_title('Pose Y Error')
        ax[1,1].set_ylabel('(m)')
        
        aw[2,1].plot(np.subtract(posz,posz3),'g')
        aw[2,1].set_title('Pose Z Error')
        aw[2,1].set_ylabel('(m)')
        
        
        
        

        plt.tight_layout()

        plt.pause(5)
        plt.close()
                
        

