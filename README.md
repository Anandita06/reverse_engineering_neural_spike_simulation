# Reverse-Engineering Neural Spike Recording — Team 3

This project analyzes a 10-second neural recording and uses its estimated drift, noise, spike shape, and activity rates to generate simulated recordings of different durations. It then compares the simulations with the original data.

**Authors:** Miloserdov Daniil, Anandita Banerjee, Mateo Maya Martinez

<img src="plots/pipeline_summary.jpeg" alt="Pipeline Summary" width="650">

See [Student_Report.md](https://github.com/Anandita06/reverse_engineering_neural_spike_simulation/blob/main/Student_Report.md) for the full analysis and report.

## Team 3 reconstruction and evaluator

The reconstruction targets `data/spike_data.npz` (10,000 samples at 1000 Hz; random seed 42). The fitted model parameters and estimation method are documented in the student report. For the inter-team reconstruction, `reconstruction` contains only the structured signal (`drift + spike_signal`); the generated Gaussian noise is stored separately in `noise`. The estimated noise standard deviation is 0.2986, measured outside a 12 ms pre-spike and 26 ms post-spike window.

Run the evaluator from the repository root with:

```python
from evaluate_team import evaluate

results = evaluate("data/spike_data.npz", "team_3_reconstruction.npz")
```

## Project layout

- `src/`: shared analysis functions and standalone validation script
- `scripts/`: preprocessing, simulation, and comparison pipeline
- `data/`: original recording and generated simulation files
- `results/`: comparison table
- `plots/`: generated figures and overview images

Install the required Python packages from the repository root:

```powershell
python -m pip install -r requirements.txt
```

Run commands from the repository root:

```powershell
python -i .\scripts\01_preprocessing_and_analysis.py
```

When the first stage finishes and shows the `>>>` prompt, run the simulation and comparison stage in that same session:

```python
exec(open("scripts/02_simulation_and_comparison.py", encoding="utf-8").read())
```

Run the independent measurement check from the repository root:

```powershell
python -m src.testing_module
```
