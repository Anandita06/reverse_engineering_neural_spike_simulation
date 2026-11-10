# 1. Load the original recording

from pathlib import Path

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.ndimage import median_filter, gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import ks_2samp
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis_utils import exponential, sine

seed = 42
plot_dir = PROJECT_ROOT / 'plots'
plot_dir.mkdir(exist_ok=True)


def save_and_show_plot(filename):
    plt.savefig(plot_dir / filename, dpi=160, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(0.001)

data_path = PROJECT_ROOT / 'data' / 'spike_data.npz'
if not data_path.exists():
    raise FileNotFoundError(
        f"Data file not found at {data_path}. "
        'Place spike_data.npz in the repository data folder.'
    )

with np.load(data_path, allow_pickle=False) as data:
    t = data['t']
    raw = data['raw_data']
fs = int(round(1 / (t[1] - t[0])))
duration_original = len(t) / fs

print('Samples:', len(t), '| Sampling rate:', fs, 'Hz', '| Duration:', duration_original, 's')
plt.figure(figsize=(12, 3))
plt.plot(t, raw, linewidth=0.6)
plt.title('Original recording')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (a.u.)')
save_and_show_plot('original_recording.png')


# 2. Drift: try a sine wave with a period of 10 seconds and an amplitude of 0.5

# The 301-ms median window removes most sharp spikes.
baseline = median_filter(raw, size=301, mode='reflect')
baseline = gaussian_filter1d(baseline, sigma=120, mode='reflect')

# We give reasonable starting values after looking at the graph.
drift_params, _ = curve_fit(
    sine, t[::5], baseline[::5],
    p0=[0, 0.6, 0.09, 0.3],
    bounds=([-1, 0, 0.055, -np.pi], [1, 2, 0.15, np.pi]),
    maxfev=10000
)
c, drift_A, drift_f, phase = drift_params
drift = sine(t, *drift_params)
print(f'Offset = {c:.3f}, amplitude = {drift_A:.3f}, frequency = {drift_f:.4f} Hz')
print(f'Period = {1/drift_f:.2f} s, phase = {phase:.3f} rad')

plt.figure(figsize=(12, 3))
plt.plot(t, raw, alpha=0.4, lw=0.5, label='Original')
plt.plot(t, baseline, lw=2, label='Estimated baseline')
plt.plot(t, drift, lw=2, label='Fitted sine')
plt.legend()
plt.xlabel('Time (s)')
save_and_show_plot('drift_fit.png')


# 3. Find spikes

peaks, _ = find_peaks(raw - baseline, height=2.4, prominence=2.0, distance=7)
print('Clear spikes:', len(peaks), '| Mean rate:', len(peaks)/duration_original, 'Hz')

plt.figure(figsize=(12, 3))
plt.plot(t, raw, lw=0.6)
plt.scatter(t[peaks], raw[peaks], s=14, color='red', label='Peaks')
plt.legend()
plt.xlabel('Time (s)')
save_and_show_plot('detected_spikes.png')


# 4. Background noise

keep = np.ones(len(raw), dtype=bool)
for p in peaks:
    keep[max(0, p-12):min(len(raw), p+26)] = False

noise_original = (raw - drift)[keep]
noise_sigma = np.std(noise_original, ddof=1)
print('Noise standard deviation:', round(noise_sigma, 4))

plt.figure(figsize=(7, 3))
plt.hist(noise_original, bins=65, density=True)
plt.title('Original noise (spikes removed)')
plt.xlabel('Residual voltage')
save_and_show_plot('noise_distribution.png')


# 5. Shape of one spike

segments = np.array([(raw - drift)[p:p+9] for p in peaks if p+9 <= len(raw)])
mean_spike = segments.mean(axis=0)
spike_t = np.arange(9) / fs

spike_params, _ = curve_fit(
    exponential, spike_t, mean_spike,
    p0=[5, 0.003], bounds=([1, 0.0005], [20, 0.02])
)
spike_A, tau = spike_params
print(f'Spike amplitude = {spike_A:.3f}, tau = {tau*1000:.3f} ms')

plt.figure(figsize=(6, 3))
plt.plot(spike_t*1000, mean_spike, 'o', label='Mean real spike')
plt.plot(spike_t*1000, exponential(spike_t, *spike_params), label='Exponential fit')
plt.xlabel('Time after peak (ms)')
plt.ylabel('Amplitude above baseline')
plt.legend()
save_and_show_plot('spike_shape.png')

# 6. Activity window and Poisson spikes

window_s = 1.0
counts, _ = np.histogram(t[peaks], bins=np.arange(0, 11, 1))
avg_rate = len(peaks) / duration_original
v = np.var(counts, ddof=1)
m = np.mean(counts)
shrink = np.sqrt(max(0, v - m) / v) if v > 0 else 0
rates = avg_rate + shrink * (counts - avg_rate)

print('Counts in the original windows:', counts.tolist())
print('Shrinkage factor:', round(shrink, 3))
print('Estimated rates (Hz):', np.round(rates, 2).tolist())

x = np.arange(10)
plt.figure(figsize=(10, 3))
plt.bar(x - 0.2, counts, width=0.4, label='Observed counts')
plt.bar(x + 0.2, rates, width=0.4, label='Estimated rates (Hz)')
plt.xticks(x, [f'{i}-{i+1}' for i in x])
plt.xlabel('Window (s)')
plt.legend()
save_and_show_plot('activity_rates.png')