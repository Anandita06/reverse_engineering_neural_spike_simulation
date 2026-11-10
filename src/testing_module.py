from pathlib import Path

import numpy as np

from .analysis_utils import measure_recording


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / 'data' / 'spike_data.npz'
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    with np.load(data_path, allow_pickle=False) as data:
        time = data['t']
        voltage = data['raw_data']

    if len(time) < 2:
        raise ValueError("The recording needs at least two time samples.")

    sampling_rate_hz = int(round(1 / (time[1] - time[0])))
    original = measure_recording(time, voltage, sampling_rate_hz)
    print(
        f"Original (independent test): {len(original['spikes'])} spikes, "
        f"{original['rate']:.2f} Hz"
    )


if __name__ == '__main__':
    main()
