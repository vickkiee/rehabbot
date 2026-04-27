from gymnasium.envs.registration import register

register(
    id='rehabbot/pid-control',
    entry_point='rehabbot.envs.pid_control:RehabEnv',
    max_episode_steps=100,
)

register(
    id='rehabbot/rehab-reach-v0',
    entry_point='rehabbot.envs.rehab_reach_v0:RehabEnv',
    max_episode_steps=1000,
)

register(
    id='rehabbot/rehab-reach-l-v0',
    entry_point='rehabbot.envs.rehab_reach_l_v0:RehabEnv',
    max_episode_steps=1000,
)

register(
    id='rehabbot/rehab-reach-m-v0',
    entry_point='rehabbot.envs.rehab_reach_m_v0:RehabEnv',
    max_episode_steps=1000,
)

register(
    id='rehabbot/rehab-hip-v0',
    entry_point='rehabbot.envs.rehab_hip_v0:RehabEnv',
    max_episode_steps=1000,
)

register(
    id='rehabbot/rehab-knee-v0',
    entry_point='rehabbot.envs.rehab_knee_v0:RehabEnv',
    max_episode_steps=1000,
)
