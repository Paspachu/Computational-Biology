#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
import pandas as pd

#%%
def run_simulation(numEx, numIn, time_max, param_a, param_b):
    totalNum = numEx + numIn

    r = stats.uniform.rvs(size=(totalNum))

    # Param a
    scale = np.ones(totalNum)
    # Param b
    uSens = np.ones(totalNum)
    # Param c
    reset = np.ones(totalNum)
    # Param d
    uReset = np.ones(totalNum)

    # Param a for excitatory neurons
    scale[0 : numEx] *= param_a
    # Param a for inhibitory neurons
    scale[numEx : ] *= 0.02 + 0.08 * r[numEx : ]

    # Param b for excitatory neurons
    uSens[0 : numEx] *= param_b
    # Param b for inhibitory neurons
    uSens[numEx : ] *= 0.25 - 0.05 * r[numEx : ]

    # Param c for excitatory neurons
    reset[0 : numEx] *= -65 + 15 * r[0 : numEx] ** 2
    # Param c for inhibitory neurons
    reset[numEx : ] *= -65

    # Param d for excitatory neurons
    uReset[0 : numEx] *= 8 - 6 * r[0 : numEx] ** 2
    # Param d for inhibitory neurons
    uReset[numEx : ] *= 2

    # Fully connected network
    synapses = np.ones((totalNum, totalNum))
    synapses[0 : numEx] *= 0.5 * stats.uniform.rvs(size=(numEx,totalNum))
    synapses[numEx : ] *= -stats.uniform.rvs(size=(numIn,totalNum))

    # Initial v for every neuron
    voltage = np.full(totalNum, -65.0)
    # Initial u for every neuron
    recovery = np.multiply( voltage,  uSens)
    # Initial thalamic input
    input = np.zeros(totalNum)

    # Run the simulation
    spikes_per_t = []
    spikes = []
    times = []
    for t in range(time_max):
        input[0 : numEx] = 5.0 * stats.norm.rvs(size=numEx)
        input[numEx : ] = 2.0 * stats.norm.rvs(size=numIn)

        pSpikes = np.where(voltage >= 30.0)[0]
        input += synapses[pSpikes, :].sum(axis=0)

        # if v >= 30 mV, v = c
        voltage[pSpikes] = reset[pSpikes]
        # if v >= 30 mV, u += d
        recovery[pSpikes] += uReset[pSpikes]

        # v += 0.04v^2 + 5v + 140 - u + I
        voltage += 0.5 * (0.04 * voltage ** 2 +
                        5.0 * voltage + 140 - recovery + input)
        voltage += 0.5 * (0.04 * voltage ** 2 +
                        5.0 * voltage + 140 - recovery + input)
        # u += a(bv - u)
        recovery += scale * (voltage * uSens - recovery)

        voltage[np.where(voltage >= 30.0)] = 30.0

        spikes_per_t.append(len(pSpikes))
        for n in pSpikes:
            spikes.append(n)
            times.append(t)

    # Obtain the last second of the simulation
    times_nparray = np.array(times)
    spikes_nparray = np.array(spikes)
    index = np.where(times_nparray >= 29000)[0][0]
    spikes_train = np.array(spikes_per_t[-1000:])

    return times_nparray[index:], spikes_nparray[index:], spikes_train

def plot_neuron_spiking(times, spikes, numEx, spikes_per_t):
    fig = plt.figure(figsize=(6,6))
    ax1 = fig.add_subplot(111)

    # Plot of spikes of each neuron over time
    ax1.plot(times, spikes, ',k')
    xl, xr = ax1.get_xlim()
    yb, yt = ax1.get_ylim()
    ax1.set_aspect(abs((xr - xl) / (yb - yt))  * 1.0)
    ax1.axhline(color='r', y=numEx - 0.5, xmax=len(spikes_per_t))
    plt.show()

    # Plot of total spikes of 1000 neurons over time
    plt.figure()
    plt.plot(range(len(spikes_per_t)), spikes_per_t)
    plt.show()

def get_power_for_each_bandwidth(numEx, numIn, time_max, param_a, param_b, Fs, trials, plot):
    total_ps = np.zeros(501)

    for i in range(trials):
        # Run simulation and return results after 29000ms
        _, _, spikes_per_t = run_simulation(numEx, numIn, time_max, param_a, param_b)
        # Get the power spectrum of the spikes
        f, Pxx_spec = signal.welch(spikes_per_t, Fs, nperseg = 1 * Fs, scaling='spectrum')
        # Sum the power spectrum
        total_ps += np.sqrt(Pxx_spec)

    total_ps /= trials

    if plot:
        boundary = 15
        plt.plot(f, total_ps)
        plt.xlim([1, boundary])
        plt.xticks(np.arange(0, boundary + 1, 1))
        plt.show()

    delta = np.sum(total_ps[1:4])
    theta = np.sum(total_ps[4:8])
    alpha = np.sum(total_ps[8:13])
    beta1 = np.sum(total_ps[13:19])
    beta2 = np.sum(total_ps[19:22])
    beta3 = np.sum(total_ps[22:31])
    gamma = np.sum(total_ps[31:51])
    full = np.sum(total_ps[1:71])

    return np.array([delta, theta, alpha, beta1, beta2, beta3, gamma, full])

#%%
# Totle simulation time
time_max = 30000     # in ms
# Frequency
Fs = 1000
# Number of trials
trials = 10

#%%
# Control group
numEx = 800
numIn = 200
param_a = 0.02
param_b = 0.2
control = get_power_for_each_bandwidth(numEx, numIn, time_max, param_a, param_b, Fs, trials, True)
control

#%%
# Decreasing number of excitatory neurons
percent_ds = np.arange(0.0075, 0.046, 0.0025)
numExs = np.array([int(800 * (1 - d)) for d in percent_ds])
numIn = 200
param_a = 0.02
param_b = 0.2
numEx_dec = []

for numEx in numExs:
    result = get_power_for_each_bandwidth(numEx, numIn, time_max, param_a, param_b, Fs, trials, True)
    numEx_dec.append(result)
numEx_dec = np.array(numEx_dec)

#%%
min_numEx_dec = np.min(numEx_dec, axis = 0)
decrease_percentage = np.zeros(len(control))
for i in range(len(control)):
    decrease_percentage[i] = 100 * (control[i] - min_numEx_dec[i]) / control[i]

case_study1 = np.array([control, min_numEx_dec, decrease_percentage])
table2 = pd.DataFrame(case_study1, columns=['delta', 'theta', 'alpha', 'beta1', 'beta2', 'beta3', 'gamma', 'full'])
table2

#%%
# Decreasing parameter b
param_bs = np.arange(0.195, 0.1996, 0.0005)
numEx = 800
numIn = 200
param_a = 0.02
param_b_dec = []

for param_b in param_bs:
    result = get_power_for_each_bandwidth(numEx, numIn, time_max, param_a, param_b, Fs, trials, True)
    param_b_dec.append(result)
param_b_dec = np.array(param_b_dec)

#%%
min_param_b_dec = np.min(param_b_dec, axis = 0)
decrease_percentage = np.zeros(len(control))
for i in range(len(control)):
    decrease_percentage[i] = 100 * (control[i] - min_param_b_dec[i]) / control[i]

case_study2 = np.array([control, min_param_b_dec, decrease_percentage])
table3 = pd.DataFrame(case_study2, columns=['delta', 'theta', 'alpha', 'beta1', 'beta2', 'beta3', 'gamma', 'full'])
table3

#%%
# Decreasing parameter a
param_as = np.arange(0.0195, 0.01996, 0.00005)
numEx = 800
numIn = 200
param_a = 0.2
param_a_dec = []

for param_a in param_as:
    result = get_power_for_each_bandwidth(numEx, numIn, time_max, param_a, param_b, Fs, trials, True)
    param_a_dec.append(result)
param_a_dec = np.array(param_a_dec)

#%%
min_param_a_dec = np.min(param_a_dec, axis = 0)
decrease_percentage = np.zeros(len(control))
for i in range(len(control)):
    decrease_percentage[i] = 100 * (control[i] - min_param_a_dec[i]) / control[i]

case_study3 = np.array([control, min_param_a_dec, decrease_percentage])
table4 = pd.DataFrame(case_study3, columns=['delta', 'theta', 'alpha', 'beta1', 'beta2', 'beta3', 'gamma', 'full'])
table4

#%%
# # Total simulation time
# time_max = 30000     # in ms
# Fs = 1000
# # Control group
# numEx = 800
# numIn = 200
# param_a = 0.02
# param_b = 0.2

# times, spikes, spikes_per_t = run_simulation(numEx, numIn, time_max, param_a, param_b)

# fig = plt.figure(figsize=(6,6))
# ax1 = fig.add_subplot(111)

# # Plot of spikes of each neuron over time
# ax1.plot(times, spikes, ',k')
# xl, xr = ax1.get_xlim()
# yb, yt = ax1.get_ylim()
# ax1.set_aspect(abs((xr - xl) / (yb - yt))  * 1.0)
# ax1.axhline(color='r', y=numEx - 0.5, xmax=time_max)
# ax1.set_xlabel('Time (ms)')
# ax1.set_ylabel('Neuron Number')
# ax1.set_title('Spikes of Neurons over Last 1 Second')
# plt.show()

# # Plot of total spikes of 1000 neurons over time
# plt.figure()
# plt.plot(range(29000, 29000 + len(spikes_per_t)), spikes_per_t)
# plt.xlabel('Time (ms)')
# plt.ylabel('Total Number of Spiked Neurons')
# plt.title('Total Number of Neurons Spiked during Last 1 Second')
# plt.show()

# # Plot power spectrum
# f, Pxx_spec = signal.welch(spikes_per_t, Fs, nperseg = 1 * Fs, scaling='spectrum')
# plt.figure()
# plt.plot(f, np.sqrt(Pxx_spec))
# plt.xlim([1, 70])
# plt.xticks(np.arange(0, 71, 5))
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Power (V RMS)')
# plt.title('Power Spectrum of Frequency Bands')
# plt.show()

# %%
