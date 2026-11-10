# 7. Complete simulation

import sys
from pathlib import Path

SCRIPT_DIR = (
    Path(__file__).resolve().parent
    if '__file__' in globals()
    else Path.cwd() / 'scripts'
)
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis_utils import exponential, measure_recording, sine

data_dir = PROJECT_ROOT / 'data'
results_dir = PROJECT_ROOT / 'results'
plot_dir = PROJECT_ROOT / 'plots'
results_dir.mkdir(exist_ok=True)
plot_dir.mkdir(exist_ok=True)

def make_spike_times(duration, rng):
    all_times = []
    rate_order = rates.copy()
    n_windows = int(np.ceil(duration / window_s))

    for i in range(n_windows):
        # Keep first 10 seconds in the fitted order; shuffle later blocks.
        if i > 0 and i % len(rates) == 0:
            rate_order = rng.permutation(rates)

        start = i * window_s
        end = min((i + 1) * window_s, duration)
        rate = rate_order[i % len(rates)]
        n = rng.poisson(rate * (end - start))
        all_times.extend(rng.uniform(start, end, n))

    all_times = np.sort(all_times)
    accepted = []
    for tm in all_times:
        if not accepted or tm - accepted[-1] >= 0.005:
            accepted.append(tm)
    return np.array(accepted)


def simulate(duration=10, seed=42):
    n = int(round(duration * fs))
    new_t = np.arange(n) / fs

    # Separate random generators keep the results reproducible.
    rng_events = np.random.default_rng(seed)
    rng_noise = np.random.default_rng(seed + 1)

    new_drift = sine(new_t, *drift_params)
    new_noise = rng_noise.normal(0, noise_sigma, n)
    signal = new_drift + new_noise

    times = make_spike_times(duration, rng_events)
    shape_t = np.arange(int(np.ceil(10 * tau * fs))) / fs
    shape = exponential(shape_t, spike_A, tau)

    for tm in times:
        i = int(np.floor(tm * fs))
        if i >= n:
            continue
        k = min(len(shape), n - i)
        signal[i:i+k] += shape[:k]

    return new_t.astype(float), signal.astype(float), times

# Generate the requested durations. All files have only t and raw_data.
for seconds in [5, 10, 50, 100]:
    tt, yy, events = simulate(seconds, seed=seed)
    np.savez_compressed(data_dir / f'simulated_{seconds}s.npz', t=tt, raw_data=yy)
    print(f'{seconds:3d} s: {len(tt):6d} samples, {len(events):3d} generated spikes, '
          f'{len(events)/seconds:.2f} Hz')

# Quick reproducibility checks
short_t, short_y, _ = simulate(5, seed=42)
long_t, long_y, _ = simulate(100, seed=42)
again_t, again_y, _ = simulate(5, seed=42)
assert np.array_equal(short_y, long_y[:len(short_y)])
assert np.array_equal(short_y, again_y)
print('Checks passed: same seed -> same output, and the 5-s prefix is unchanged.')

# 8. Compare the simulated recordings with the original

def compare_records(ref, sim):
    n = min(len(ref['voltage']), len(sim['voltage']), 10*fs)
    drift_error = np.sqrt(np.mean((ref['baseline'][:n] - sim['baseline'][:n])**2))
    noise_ks = ks_2samp(ref['noise'], sim['noise']).statistic
    isi_ks = ks_2samp(ref['isi'], sim['isi']).statistic

    # Each score has: weight, difference, tolerance.
    # Tolerances are our own reasonable choices, not values from the teacher.
    rate_se = np.sqrt(max(ref['rate'], .01)/(len(ref['time'])/fs) +
                      max(sim['rate'], .01)/(len(sim['time'])/fs))
    parts = [
        (20, abs(ref['rate']-sim['rate']), max(.6, 1.5*rate_se)),
        (20, drift_error, .12),
        (12, abs(ref['noise_std']-sim['noise_std']), max(.015, .1*ref['noise_std'])),
        (8, noise_ks, .06),
        (10, abs(ref['amp']-sim['amp']), .4),
        (10, abs(ref['tau_ms']-sim['tau_ms']), .8),
        (10, abs(ref['fano']-sim['fano']), .8),
        (5, isi_ks, .25),
    ]
    # Extra 5 points: lag-1 correlation in noise (white noise should be uncorrelated).
    def lag1(d):
        return np.corrcoef(d['noise'][:-1], d['noise'][1:])[0, 1]
    parts.append((5, abs(lag1(ref)-lag1(sim)), .07))

    score = sum(w * np.exp(-.5*(d/tol)**2) if np.isfinite(d) else 0
                for w, d, tol in parts)
    return score, drift_error, noise_ks, isi_ks

original = measure_recording(t, raw, fs)

print('Duration | Spikes | Rate (Hz) | Noise SD | Fano | Similarity / 100')
results = {}
for seconds in [5, 10, 50, 100]:
    with np.load(data_dir / f'simulated_{seconds}s.npz', allow_pickle=False) as arr:
        test = measure_recording(arr['t'], arr['raw_data'], fs)
    score, drift_err, noise_ks, isi_ks = compare_records(original, test)
    results[seconds] = (test, score)
    print(f'{seconds:>7} s | {len(test["spikes"]):>6} | {test["rate"]:>9.2f} | '
          f'{test["noise_std"]:>8.3f} | {test["fano"]:>4.2f} | {score:>6.1f}')

# Basic checks of all generated file structures
for seconds in [5, 10, 50, 100]:
    with np.load(data_dir / f'simulated_{seconds}s.npz', allow_pickle=False) as f:
        assert set(f.files) == {'t', 'raw_data'}
        assert f['t'].shape == f['raw_data'].shape == (seconds*fs,)
        assert np.allclose(np.diff(f['t']), 1/fs)
print('File format checks passed.')


# Plots of originl and simulated recordings

sim10 = results[10][0]
fig, ax = plt.subplots(2, 2, figsize=(12, 7))
ax[0, 0].plot(t, raw, lw=.6)
ax[0, 0].plot(t, original['baseline'], lw=1.5)
ax[0, 0].set_title('Original 10 s')
ax[0, 1].plot(sim10['time'], sim10['voltage'], lw=.6)
ax[0, 1].plot(sim10['time'], sim10['baseline'], lw=1.5)
ax[0, 1].set_title('Simulated 10 s')
ax[1, 0].plot(np.arange(9), original['waveform'], 'o-', label='Original')
ax[1, 0].plot(np.arange(9), sim10['waveform'], 'o-', label='Simulated')
ax[1, 0].set_title('Mean spike shape')
ax[1, 0].set_xlabel('Time after detection (ms)')
ax[1, 0].legend()
ax[1, 1].hist(original['noise'], bins=60, density=True, alpha=.5, label='Original')
ax[1, 1].hist(sim10['noise'], bins=60, density=True, alpha=.5, label='Simulated')
ax[1, 1].set_title('Noise distribution')
ax[1, 1].legend()
for a in [ax[0,0], ax[0,1]]:
    a.set_xlabel('Time (s)')
    a.set_ylabel('Voltage (a.u.)')
plt.tight_layout()
plt.savefig(plot_dir / 'comparison_10s.png', dpi=160, bbox_inches='tight')
plt.show(block=False)
plt.pause(0.001)

# 9. Full comparison of all durations

import numpy as np
import pandas as pd

# 9.1. Original and simulated recordings
records = {
    'Original': original,
    '10 s': results[10][0],
    '50 s': results[50][0],
    '100 s': results[100][0]
}

# 9.2. Calculate parameters for each recording
def get_metrics(d):
    voltage = d['voltage']
    baseline = d['baseline']
    isi = d['isi'] * 1000  # seconds -> milliseconds

    return {
        'Samples': len(voltage),
        'Duration (s)': len(voltage) / fs,
        'Sampling rate (Hz)': fs,

        'Min voltage': np.min(voltage),
        'Max voltage': np.max(voltage),
        'Mean voltage': np.mean(voltage),
        'Median voltage': np.median(voltage),
        'Voltage SD': np.std(voltage),

        'Baseline mean': np.mean(baseline),
        'Baseline range': np.ptp(baseline),

        'Detected spikes': len(d['spikes']),
        'Firing rate (Hz)': d['rate'],
        'Spike amplitude': d['amp'],
        'Spike decay tau (ms)': d['tau_ms'],

        'Noise SD': d['noise_std'],

        'Mean ISI (ms)': np.mean(isi) if len(isi) else np.nan,
        'Median ISI (ms)': np.median(isi) if len(isi) else np.nan
    }


# 9.3. Create the table
comparison = pd.DataFrame({
    name: get_metrics(data)
    for name, data in records.items()
})


# 9.4. Add baseline RMSE
comparison.loc['Baseline RMSE', 'Original'] = 0

for seconds in [10, 50, 100]:
    sim = results[seconds][0]

    n = min(
        len(original['baseline']),
        len(sim['baseline'])
    )

    rmse = np.sqrt(np.mean(
        (original['baseline'][:n] - sim['baseline'][:n]) ** 2
    ))

    comparison.loc['Baseline RMSE', f'{seconds} s'] = rmse


# 9.5. Add our similarity scores
comparison.loc['Similarity score / 100', 'Original'] = 100

for seconds in [10, 50, 100]:
    comparison.loc[
        'Similarity score / 100',
        f'{seconds} s'
    ] = results[seconds][1]


# 9.6. Choose metrics for percentage closeness
# We skip duration, sample count and values near zero.

closeness_metrics = [
    'Voltage SD',
    'Baseline range',
    'Firing rate (Hz)',
    'Spike amplitude',
    'Spike decay tau (ms)',
    'Noise SD',
    'Mean ISI (ms)',
    'Median ISI (ms)'
]


# 9.7. Calculate closeness (%)
def closeness(original_value, simulated_value):
    if not np.isfinite(original_value):
        return np.nan

    if not np.isfinite(simulated_value):
        return np.nan

    if original_value == 0:
        return np.nan

    difference = abs(simulated_value - original_value)

    score = 100 * (1 - difference / abs(original_value))

    return np.clip(score, 0, 100)


for seconds in [10, 50, 100]:
    column = f'{seconds} s'
    close_column = f'{seconds} s closeness (%)'

    comparison[close_column] = np.nan

    for metric in closeness_metrics:
        comparison.loc[metric, close_column] = closeness(
            comparison.loc[metric, 'Original'],
            comparison.loc[metric, column]
        )


# 9.8. Arrange columns
comparison = comparison[
    [
        'Original',
        '10 s',
        '10 s closeness (%)',
        '50 s',
        '50 s closeness (%)',
        '100 s',
        '100 s closeness (%)'
    ]
]


# 9.9 Display table
print(comparison.to_string(na_rep='-'))

# 9.10. Save as CSV
comparison.to_csv(results_dir / 'final_comparison.csv', index_label='Metric', na_rep='-')

print(f'Saved: {results_dir / "final_comparison.csv"}')