![Pipeline Summary](plots/pipeline_summary.jpeg)

# Project overview

See [Student_Report.md](Student_Report.md) for the full analysis and report.

## Project layout

- `src/`: shared analysis functions and standalone validation script
- `scripts/`: preprocessing, simulation, and comparison pipeline
- `data/`: original recording and generated simulation files
- `results/`: comparison table
- `plots/`: generated figures and overview images

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
