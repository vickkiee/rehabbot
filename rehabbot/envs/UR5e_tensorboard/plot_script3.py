import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================
# Main function
# =====================================
def main():

    
    plt.rcParams.update({
        'font.size': 8,          # Base font size
        'axes.titlesize': 8,     # Title font size
        'axes.labelsize': 8,     # Axis label font size
        'xtick.labelsize': 7,    # X-tick label size
        'ytick.labelsize': 7,    # Y-tick label size
        'legend.fontsize': 7,    # Legend font size
        'lines.linewidth': 1.2,  # Line width
        'figure.dpi': 600,       # High-resolution output
        'figure.figsize': (3.5, 2.5),  # Single-column size (inches)
        'grid.alpha': 0.3        # Grid transparency
    })
    
    SAC_len_R = []
    SAC_rew_R = []
    SAC_len_L = []
    SAC_rew_L = []
    TD3_len_R = []
    TD3_rew_R = []
    TD3_len_L = []
    TD3_rew_L = []
    PPO_len_R = []
    PPO_rew_R = []
    PPO_len_L = []
    PPO_rew_L = []
        
    SAC_len_R = pd.read_csv('SAC-1.5M-R-2.5_ep_len_mean.csv')  
    SAC_rew_R = pd.read_csv('SAC-1.5M-R-2.5_ep_rew_mean.csv')
    
    SAC_len_L = pd.read_csv('SAC-1.5M-L-2.5_ep_len_mean.csv')  
    SAC_rew_L = pd.read_csv('SAC-1.5M-L-2.5_ep_rew_mean.csv')
    
    TD3_len_R = pd.read_csv('TD3-1.5M-R-2.5_ep_len_mean.csv')  
    TD3_rew_R = pd.read_csv('TD3-1.5M-R-2.5_ep_rew_mean.csv')
    
    TD3_len_L = pd.read_csv('TD3-1.5M-L-2.5_ep_len_mean.csv')  
    TD3_rew_L = pd.read_csv('TD3-1.5M-L-2.5_ep_rew_mean.csv')
    
    PPO_len_R = pd.read_csv('PPO-1.5M-R-2.5_ep_len_mean.csv')  
    PPO_rew_R = pd.read_csv('PPO-1.5M-R-2.5_ep_rew_mean.csv')
    
    PPO_len_L = pd.read_csv('PPO-1.5M-L-2.5_ep_len_mean.csv')  
    PPO_rew_L = pd.read_csv('PPO-1.5M-L-2.5_ep_rew_mean.csv')
    

    
    SAC_len_R_step = SAC_len_R['Step'].values
    SAC_rew_R_step = SAC_rew_R['Step'].values
    SAC_len_L_step = SAC_len_L['Step'].values
    SAC_rew_L_step = SAC_rew_L['Step'].values
    
    TD3_len_R_step = TD3_len_R['Step'].values
    TD3_rew_R_step = TD3_rew_R['Step'].values
    TD3_len_L_step = TD3_len_L['Step'].values
    TD3_rew_L_step = TD3_rew_L['Step'].values
    
    PPO_len_R_step = PPO_len_R['Step'].values
    PPO_rew_R_step = PPO_rew_R['Step'].values
    PPO_len_L_step = PPO_len_L['Step'].values
    PPO_rew_L_step = PPO_rew_L['Step'].values
    
    
    
    SAC_len_R_value = SAC_len_R['Value'].values
    SAC_rew_R_value = SAC_rew_R['Value'].values
    SAC_len_L_value = SAC_len_L['Value'].values
    SAC_rew_L_value = SAC_rew_L['Value'].values
    
    TD3_len_R_value = TD3_len_R['Value'].values
    TD3_rew_R_value = TD3_rew_R['Value'].values
    TD3_len_L_value = TD3_len_L['Value'].values
    TD3_rew_L_value = TD3_rew_L['Value'].values
    
    PPO_len_R_value = PPO_len_R['Value'].values
    PPO_rew_R_value = PPO_rew_R['Value'].values
    PPO_len_L_value = PPO_len_L['Value'].values
    PPO_rew_L_value = PPO_rew_L['Value'].values
    
    def create_smooth(arr):
        alpha=0.9
        smoothed = np.zeros_like(arr)
        smoothed[0] = arr[0]
        for i in range(1, len(arr)):
            smoothed[i] = alpha * smoothed[i-1] + (1 - alpha) * arr[i]
        return smoothed

    
    SAC_len_R_value = create_smooth(SAC_len_R_value)
    SAC_rew_R_value = create_smooth(SAC_rew_R_value)
    SAC_len_L_value = create_smooth(SAC_len_L_value)
    SAC_rew_L_value = create_smooth(SAC_rew_L_value)
    
    TD3_len_R_value = create_smooth(TD3_len_R_value)
    TD3_rew_R_value = create_smooth(TD3_rew_R_value)
    TD3_len_L_value = create_smooth(TD3_len_L_value)
    TD3_rew_L_value = create_smooth(TD3_rew_L_value)
    
    PPO_len_R_value = create_smooth(PPO_len_R_value)
    PPO_rew_R_value = create_smooth(PPO_rew_R_value)
    PPO_len_L_value = create_smooth(PPO_len_L_value)
    PPO_rew_L_value = create_smooth(PPO_rew_L_value)
     
    #print(SAC_len_R_step)
    #print(SAC_len_R_value)
    #print(TD3_len_L_step)
    #print(TD3_len_L_value)
    #print(SAC_rew_L_step)
    #print(SAC_rew_L_value)
        
    _, az = plt.subplots(1, 2, sharex=True, figsize=(10, 6))
    
        
    # Plot Episode mean length
    az[0].plot(SAC_len_R_step, SAC_len_R_value, color="#1f77b4", label="SAC Episode mean length (RH)", alpha=0.9)  
    az[0].plot(TD3_len_R_step, TD3_len_R_value, color="#ff7f0e", label="TD3 Episode mean length (RH)", alpha=0.9)
    az[0].plot(PPO_len_R_step, PPO_len_R_value, color="#2ca02c", label="PPO Episode mean length (RH)", alpha=0.9)
    
    az[1].plot(SAC_len_L_step, SAC_len_L_value, color="#1f77b4", label="SAC Episode mean length (LH)", alpha=0.9)  
    az[1].plot(TD3_len_L_step, TD3_len_L_value, color="#ff7f0e", label="TD3 Episode mean length (LH)", alpha=0.9)
    az[1].plot(PPO_len_L_step, PPO_len_L_value, color="#2ca02c", label="PPO Episode mean length (LH)", alpha=0.9)
    
    az[0].set_xlabel("Training Timesteps", labelpad=2)
    az[0].set_ylabel("Mean Episode Length", labelpad=2)
    az[0].set_title(f"RehabBot Mean Episode Length (RH)")
    az[0].legend(frameon=True, loc='best', handlelength=1.0)
    az[0].grid(True)
    az[0].ticklabel_format(axis='x', style='sci', scilimits=(5,5))
    az[0].minorticks_on()
    az[0].tick_params(axis='both', which='both', direction='in', pad=2)
    
    az[1].set_xlabel("Training Timesteps", labelpad=2)
    az[1].set_ylabel("Mean Episode Length", labelpad=2)
    az[1].set_title(f"RehabBot Mean Episode Length (LH)")
    az[1].legend(frameon=True, loc='best', handlelength=1.0)  
    az[1].grid(True)
    az[1].ticklabel_format(axis='x', style='sci', scilimits=(5,5))
    az[1].minorticks_on()
    az[1].tick_params(axis='both', which='both', direction='in', pad=2)
    
    plt.tight_layout(pad=0.5)
    plt.savefig("RehabBot_Mean_Episode_Length.pdf", format='pdf')
    plt.savefig("RehabBot_Mean_Episode_Length.png", dpi=900)
    plt.close()
    

    
    _, ay = plt.subplots(1, 2, sharex=True, figsize=(10, 6))

    # Plot Episode mean reward
    ay[0].plot(SAC_rew_R_step, SAC_rew_R_value, color="#1f77b4", label="SAC Episode mean reward (RH)", alpha=0.9)  
    ay[0].plot(TD3_rew_R_step, TD3_rew_R_value, color="#ff7f0e", label="TD3 Episode mean reward (RH)", alpha=0.9)
    ay[0].plot(PPO_rew_R_step, PPO_rew_R_value, color="#2ca02c", label="PPO Episode mean reward (RH)", alpha=0.9)
    
    ay[1].plot(SAC_rew_L_step, SAC_rew_L_value, color="#1f77b4", label="SAC Episode mean reward (LH)", alpha=0.9)  
    ay[1].plot(TD3_rew_L_step, TD3_rew_L_value, color="#ff7f0e", label="TD3 Episode mean reward (LH)", alpha=0.9)
    ay[1].plot(PPO_rew_L_step, PPO_rew_L_value, color="#2ca02c", label="PPO Episode mean reward (LH)", alpha=0.9)
    
    ay[0].set_xlabel("Training Timesteps", labelpad=2)
    ay[0].set_ylabel("Mean Episode Reward", labelpad=2)
    ay[0].set_title(f"RehabBot Mean Episode Reward (RH)")
    ay[0].legend(frameon=True, loc='best', handlelength=1.0)
    ay[0].grid(True)
    ay[0].ticklabel_format(axis='x', style='sci', scilimits=(5,5))
    ay[0].minorticks_on()
    ay[0].tick_params(axis='both', which='both', direction='in', pad=2)
    
    ay[1].set_xlabel("Training Timesteps", labelpad=2)
    ay[1].set_ylabel("Mean Episode Reward", labelpad=2)
    ay[1].set_title(f"RehabBot Mean Episode Reward (LH)")
    ay[1].legend(frameon=True, loc='best', handlelength=1.0)
    ay[1].grid(True)
    ay[1].ticklabel_format(axis='x', style='sci', scilimits=(5,5))
    ay[1].minorticks_on()
    ay[1].tick_params(axis='both', which='both', direction='in', pad=2)
    
    plt.tight_layout(pad=0.5)
    plt.savefig("RehabBot_Mean_Episode_Reward.pdf", format='pdf')
    plt.savefig("RehabBot_Mean_Episode_Reward.png", dpi=900)
    plt.close()


if __name__ == "__main__":
    main()