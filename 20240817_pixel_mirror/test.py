

#%%
import numpy as np
import matplotlib.pyplot as plt



#%%

dthetaL = 2/180*np.pi
N_L = 10
thetaLs = dthetaL * np.linspace(-N_L, N_L, 2*N_L+1)

N_in = 1000
theta_in = dthetaL * np.linspace(-N_L, N_L, N_in)
theta_out = np.zeros(N_in)

n = 1.45
for ii in range(N_in):
    ind = int(np.round(theta_in[ii]/dthetaL))
    thetaL = dthetaL * ind
    theta_out[ii] = theta_in[ii]/n + (1/n-1)*thetaL



#%%

plt.plot(theta_in, theta_out, '.-')
plt.xlabel('theta in')
plt.ylabel('theta out')

# %%
