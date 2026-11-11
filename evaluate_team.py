"""
evaluate_team.py
----------------
Inter-Team Evaluator for Reverse-Engineering Competition.

Required Interface:
    def evaluate(reference_file, candidate_file) -> dict
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks

REQUIRED_KEYS = (
    "t",
    "drift",
    "spike_signal",
    "noise",
    "reconstruction",
    "spike_indices",
)
EXPECTED_N = 10000


def _read_reference(reference_file: str | Path):
    with np.load(reference_file, allow_pickle=False) as f:
        if not {"t", "raw_data"}.issubset(f.files):
            raise ValueError("Reference file must contain 't' and 'raw_data'.")
        t = np.asarray(f["t"], float)
        raw = np.asarray(f["raw_data"], float)

    if t.shape != (EXPECTED_N,) or raw.shape != (EXPECTED_N,):
        raise ValueError(f"Reference must contain exactly {EXPECTED_N} samples.")
    if not np.isfinite(t).all() or not np.isfinite(raw).all():
        raise ValueError("Reference contains NaN or Inf values.")

    dt = float(np.median(np.diff(t)))
    if dt <= 0 or not np.allclose(np.diff(t), dt, rtol=1e-3, atol=1e-8):
        raise ValueError("Reference time vector is not uniformly increasing.")

    return t, raw, 1.0 / dt


def _load_candidate(candidate_file: str | Path, ref_t: np.ndarray):
    with np.load(candidate_file, allow_pickle=False) as f:
        missing = sorted(set(REQUIRED_KEYS) - set(f.files))
        if missing:
            raise ValueError(f"Candidate file missing required arrays: {missing}")
        data = {k: np.asarray(f[k]) for k in REQUIRED_KEYS}

    for key in REQUIRED_KEYS[:-1]:  # Check all 1D (10000,) arrays
        arr = data[key]
        if arr.shape != (EXPECTED_N,):
            raise ValueError(f"Array '{key}' has shape {arr.shape}; expected ({EXPECTED_N},).")
        if not np.issubdtype(arr.dtype, np.number) or not np.isrealobj(arr):
            raise ValueError(f"Array '{key}' must be real numeric.")
        if not np.isfinite(arr).all():
            raise ValueError(f"Array '{key}' contains NaN or Inf.")

    if not np.allclose(data["t"], ref_t, rtol=0.0, atol=1e-6):
        raise ValueError("Candidate time vector does not match reference.")

    indices = data["spike_indices"]
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("spike_indices must be a 1D array of integers.")
    if np.any(indices < 0) or np.any(indices >= EXPECTED_N):
        raise ValueError("spike_indices out of bounds.")
    if len(indices) > 1 and np.any(np.diff(indices) <= 0):
        raise ValueError("spike_indices must be strictly increasing.")

    # Reconstruction check: reconstruction must be drift + spike_signal (no noise)
    if not np.allclose(
        data["reconstruction"],
        data["drift"] + data["spike_signal"],
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError("reconstruction must equal drift + spike_signal.")

    return data


def _f1_with_tolerance(reference_spikes, candidate_spikes, tolerance):
    a, b = np.asarray(reference_spikes), np.asarray(candidate_spikes)
    if len(a) == 0 and len(b) == 0:
        return 1.0, 0, 0, 0

    i = j = matched = 0
    while i < len(a) and j < len(b):
        if abs(int(a[i]) - int(b[j])) <= tolerance:
            matched += 1
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1

    precision = matched / len(b) if len(b) else 0.0
    recall = matched / len(a) if len(a) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return float(f1), int(matched), int(len(b)), int(len(a))


def _score_close(error, tolerance):
    return float(np.exp(-0.5 * (max(0.0, error) / max(tolerance, 1e-9)) ** 2))


def evaluate(reference_file: str | Path, candidate_file: str | Path) -> dict:
    """
    Inputs:
        reference_file: path to original spike_data.npz
        candidate_file: path to another team's reconstruction npz file

    Output:
        dictionary containing:
        - validity checks
        - numerical evaluation metrics
        - final score (0 to 100)
    """
    result = {
        "candidate": str(candidate_file),
        "valid": False,
        "validity_checks": [],
        "metrics": {},
        "final_score": 0.0,
    }

    try:
        t, raw, fs = _read_reference(reference_file)
        d = _load_candidate(candidate_file, t)
        result["validity_checks"].append("Required keys, shapes, and types: OK")
        result["validity_checks"].append("t and reconstruction = drift + spike_signal: OK")

        # Reference baseline & spike detection for evaluation
        window = max(3, int(round(0.301 * fs)))
        if window % 2 == 0:
            window += 1
        baseline = median_filter(raw, size=window, mode="reflect")
        baseline = gaussian_filter1d(baseline, sigma=0.120 * fs, mode="reflect")

        ref_peaks, _ = find_peaks(
            raw - baseline,
            height=2.4,
            prominence=2.0,
            distance=max(1, int(round(0.007 * fs))),
        )

        mask = np.ones(len(raw), dtype=bool)
        for p in ref_peaks:
            mask[
                max(0, int(p) - int(round(0.012 * fs))) : min(
                    len(raw), int(p) + int(round(0.026 * fs))
                )
            ] = False
        ref_sigma = float(np.std((raw - baseline)[mask], ddof=1)) if mask.sum() > 1 else 0.3

        # Numerical evaluations
        drift_error = float(np.sqrt(np.mean((d["drift"] - baseline) ** 2)))
        raw_rmse = float(np.sqrt(np.mean((raw - d["reconstruction"]) ** 2)))
        excess_rmse = max(0.0, raw_rmse - ref_sigma)

        f1, n_matched, n_candidate, n_reference = _f1_with_tolerance(
            ref_peaks, d["spike_indices"], tolerance=max(1, int(round(0.005 * fs)))
        )

        # Waveform evaluation
        n_wave = max(2, int(round(0.009 * fs)))
        ref_segments = [
            (raw - baseline)[int(p) : int(p) + n_wave]
            for p in ref_peaks
            if int(p) + n_wave <= EXPECTED_N
        ]
        cand_segments = [
            d["spike_signal"][int(p) : int(p) + n_wave]
            for p in d["spike_indices"]
            if int(p) + n_wave <= EXPECTED_N
        ]

        if ref_segments and cand_segments:
            mean_ref = np.mean(ref_segments, axis=0)
            mean_cand = np.mean(cand_segments, axis=0)
            waveform_rmse = float(np.sqrt(np.mean((mean_ref - mean_cand) ** 2)))
            waveform_similarity = _score_close(waveform_rmse, 0.75)
        else:
            waveform_rmse, waveform_similarity = float("nan"), 0.0

        # Noise evaluation
        noise_sd = float(np.std(d["noise"], ddof=1))
        noise_sd_rel_error = abs(noise_sd - ref_sigma) / ref_sigma
        noise_mean_error = abs(float(np.mean(d["noise"]))) / ref_sigma
        noise_lag1 = (
            float(np.corrcoef(d["noise"][:-1], d["noise"][1:])[0, 1])
            if noise_sd > 1e-12
            else 1.0
        )
        if not np.isfinite(noise_lag1):
            noise_lag1 = 1.0

        noise_score = (
            0.6 * _score_close(noise_sd_rel_error, 0.20)
            + 0.2 * _score_close(noise_mean_error, 0.15)
            + 0.2 * _score_close(abs(noise_lag1), 0.15)
        )

        points = {
            "drift": 20 * _score_close(drift_error, 0.25),
            "structured_reconstruction": 30 * _score_close(excess_rmse, 0.22),
            "spike_timing": 25 * f1,
            "spike_shape": 15 * waveform_similarity,
            "noise": 10 * noise_score,
        }

        result["valid"] = True
        result["metrics"] = {
            "reference_detected_spikes": n_reference,
            "candidate_spikes": n_candidate,
            "matched_spikes_within_5ms": n_matched,
            "spike_timing_f1": round(f1, 5),
            "drift_rmse": round(drift_error, 6),
            "raw_to_structured_rmse": round(raw_rmse, 6),
            "noise_std": round(noise_sd, 6),
            "noise_lag1_correlation": round(noise_lag1, 6),
            "component_points": {k: round(v, 3) for k, v in points.items()},
        }
        result["final_score"] = round(float(np.clip(sum(points.values()), 0, 100)), 3)

    except Exception as exc:
        result["validity_checks"].append(f"FAILED: {exc}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate team reconstruction.")
    parser.add_argument(
        "reference",
        nargs="?",
        default="data/spike_data.npz",
        help="Path to original spike_data.npz",
    )
    parser.add_argument(
        "candidate",
        nargs="?",
        default="team_3_reconstruction.npz",
        help="Path to team reconstruction npz",
    )
    args = parser.parse_args()

    res = evaluate(args.reference, args.candidate)
    print(json.dumps(res, indent=2))