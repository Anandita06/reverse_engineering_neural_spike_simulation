"""
Script 03: Create Reconstruction File for Inter-Team Evaluation
---------------------------------------------------------------
Generates `team_3_reconstruction.npz` containing all six required components:
  1. t             (10000,) - Time vector
  2. drift         (10000,) - Estimated sine drift
  3. spike_signal  (10000,) - Estimated spike waveform train
  4. noise         (10000,) - Generated noise realization (seed 42)
  5. reconstruction (10000,) - Clean structured signal (drift + spike_signal)
  6. spike_indices (n_spikes,) - Detected spike peak/starting indices
"""

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# Handle path resolution relative to repository root
SCRIPT_DIR = (
    Path(__file__).resolve().parent
    if "__file__" in globals()
    else Path.cwd() / "scripts"
)
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis_utils import exponential, sine

TEAM_ID = "3"
SEED = 42


def generate_reconstruction(team_id=TEAM_ID, seed=SEED):
    data_path = PROJECT_ROOT / "data" / "spike_data.npz"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Original recording not found at {data_path}. "
            "Please ensure spike_data.npz is in the data folder."
        )

    # Load original recording
    with np.load(data_path, allow_pickle=False) as data:
        t = np.asarray(data["t"], dtype=float)
        raw = np.asarray(data["raw_data"], dtype=float)

    fs = int(round(1 / (t[1] - t[0])))

    # 1. Estimate baseline / drift
    median_size = max(3, int(round(0.301 * fs)))
    if median_size % 2 == 0:
        median_size += 1

    baseline = median_filter(raw, size=median_size, mode="reflect")
    baseline = gaussian_filter1d(baseline, sigma=0.120 * fs, mode="reflect")

    drift_params, _ = curve_fit(
        sine,
        t[::5],
        baseline[::5],
        p0=[0.0, 0.6, 0.09, 0.3],
        bounds=([-1.0, 0.0, 0.055, -np.pi], [1.0, 2.0, 0.15, np.pi]),
        maxfev=10000,
    )
    drift = sine(t, *drift_params)

    # 2. Detect spikes
    spike_indices, _ = find_peaks(
        raw - baseline,
        height=2.4,
        prominence=2.0,
        distance=max(1, int(round(0.007 * fs))),
    )

    # 3. Fit mean exponential spike waveform
    segment_len = max(2, int(round(0.009 * fs)))
    segments = [
        (raw - drift)[p : p + segment_len]
        for p in spike_indices
        if p + segment_len <= len(raw)
    ]
    mean_spike = np.mean(np.asarray(segments), axis=0)

    spike_t = np.arange(segment_len) / fs
    (spike_A, tau), _ = curve_fit(
        exponential,
        spike_t,
        mean_spike,
        p0=[5.0, 0.003],
        bounds=([1.0, 0.0005], [20.0, 0.02]),
        maxfev=10000,
    )

    # Reconstruct continuous spike waveform signal
    spike_signal = np.zeros_like(t)
    template_t = np.arange(max(1, int(np.ceil(10.0 * tau * fs)))) / fs
    template = exponential(template_t, spike_A, tau)

    for p in spike_indices:
        k = min(len(template), len(t) - p)
        spike_signal[p : p + k] += template[:k]

    # 4. Estimate noise standard deviation & generate noise realization
    keep = np.ones(len(raw), dtype=bool)
    before = int(round(0.012 * fs))
    after = int(round(0.026 * fs))
    for p in spike_indices:
        keep[max(0, p - before) : min(len(raw), p + after)] = False

    noise_sigma = float(np.std((raw - drift)[keep], ddof=1))
    noise = np.random.default_rng(seed).normal(0.0, noise_sigma, len(raw))

    # 5. Build clean reconstruction (drift + spike_signal)
    reconstruction = drift + spike_signal

    # Save array to root directory per competition requirements
    output_path = PROJECT_ROOT / f"team_{team_id}_reconstruction.npz"
    np.savez_compressed(
        output_path,
        t=t,
        drift=drift,
        spike_signal=spike_signal,
        noise=noise,
        reconstruction=reconstruction,
        spike_indices=np.asarray(spike_indices, dtype=np.int64),
    )

    print(f"\nSuccessfully generated: {output_path}")
    print(f"  Samples: {len(t)}")
    print(f"  Detected Spikes: {len(spike_indices)}")
    print(f"  Drift Amplitude: {drift_params[1]:.4f}")
    print(f"  Spike Amplitude: {spike_A:.3f}, Tau: {tau*1000:.3f} ms")
    print(f"  Noise SD: {noise_sigma:.4f}")

    # Verify keys and shapes
    with np.load(output_path, allow_pickle=False) as f:
        print("  Arrays saved:", {key: f[key].shape for key in f.files})


if __name__ == "__main__":
    generate_reconstruction()