import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.optimize import curve_fit


def sine(time, offset, amplitude, frequency, phase):
    return offset + amplitude * np.sin(2 * np.pi * frequency * time + phase)


def exponential(time, amplitude, tau):
    return amplitude * np.exp(-time / tau)


def measure_recording(time, voltage, sampling_rate_hz):
    time = np.asarray(time, dtype=float)
    voltage = np.asarray(voltage, dtype=float)
    if time.ndim != 1 or voltage.ndim != 1 or time.size != voltage.size:
        raise ValueError("time and voltage must be one-dimensional arrays of equal length.")
    if time.size == 0:
        raise ValueError("time and voltage must not be empty.")
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive.")

    baseline = gaussian_filter1d(
        median_filter(voltage, size=301, mode='reflect'),
        sigma=120,
        mode='reflect',
    )

    local_median = median_filter(voltage, size=501, mode='reflect')
    local_mad = median_filter(abs(voltage - local_median), size=501, mode='reflect')
    threshold = local_median + 6 * 1.4826 * local_mad

    above = voltage > threshold
    starts = np.flatnonzero(above & ~np.r_[False, above[:-1]])
    spikes = []
    for index in starts:
        if not spikes or index - spikes[-1] >= 5:
            spikes.append(index)
    spikes = np.asarray(spikes, dtype=int)

    keep = np.ones(len(voltage), dtype=bool)
    for peak in spikes:
        keep[max(0, peak - 12):min(len(voltage), peak + 26)] = False
    noise = (voltage - baseline)[keep]

    residual = voltage - baseline
    pieces = np.asarray([
        residual[peak:peak + 9]
        for peak in spikes
        if peak + 9 <= len(voltage)
    ])
    if len(pieces) >= 3:
        waveform = pieces.mean(axis=0)
        fitted, _ = curve_fit(
            exponential,
            np.arange(9) / sampling_rate_hz,
            waveform,
            p0=[5, 0.003],
            bounds=([0.1, 0.0005], [20, 0.02]),
        )
    else:
        waveform = np.full(9, np.nan)
        fitted = [np.nan, np.nan]

    full_windows = int(len(time) / sampling_rate_hz)
    counts = np.histogram(time[spikes], bins=np.arange(full_windows + 1))[0]
    mean_count = np.mean(counts)
    fano = (
        np.var(counts, ddof=1) / mean_count
        if full_windows >= 3 and mean_count > 0
        else np.nan
    )

    return {
        'time': time,
        'voltage': voltage,
        'baseline': baseline,
        'spikes': spikes,
        'rate': len(spikes) / (len(time) / sampling_rate_hz),
        'noise': noise,
        'noise_std': np.std(noise, ddof=1),
        'waveform': waveform,
        'amp': fitted[0],
        'tau_ms': fitted[1] * 1000,
        'counts': counts,
        'fano': fano,
        'isi': np.diff(time[spikes]),
    }
