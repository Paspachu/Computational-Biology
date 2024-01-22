#%%
import matplotlib.pylab as plt   
import numpy as np 
import scipy.integrate as scint
import scipy.stats as stats

#%%
def f(t, x):
    V = x[0]
    u = x[1]

    I = 5.0 * stats.norm.rvs(size=1)[0]

    dVdt = 0.04 * V ** 2 + 5 * V + 140 - u + I
    dudt = 0.02 * (0.2 * V - u)

    return np.array([dVdt, dudt])

#%%
# Specify our initial condition
x0 = np.array([-65, -65 * 0.2])

# Specify trange, a time range for simulation: two element array
tstart = 0
tend = 1000
trange = np.array([tstart, tend])

# Specify tlist,  a list of time points at which want to return the solution
numt = 1000000
tlist = np.linspace(tstart, tend, numt)

# Initialize parameters
sol = scint.solve_ivp(f, trange, x0, t_eval = tlist)

# Plot the solutions
fig, axs = plt.subplots(2)

# V function
axs[0].plot(sol.t, sol.y[0, :])
axs[0].set_xticks(np.arange(tstart, tend + 1, int((tend-tstart)/2)))
axs[0].set_ylabel('V(t)')

# u function
axs[1].plot(sol.t, sol.y[1,:])
axs[1].set_xticks(np.arange(tstart, tend + 1, int((tend-tstart)/2)))
axs[1].set_ylabel('u(t)')
axs[1].set_xlabel('Time')

fig.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.2, wspace=0.2)

# %%
