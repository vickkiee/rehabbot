import numpy as np
data = np.load('evaluations.npz')
print(data)
print(data.files)

for i in range(len(data['timesteps'])):
    print(data['timesteps'][i], data['results'][i, 0:], data['ep_lengths'][i, 0:])