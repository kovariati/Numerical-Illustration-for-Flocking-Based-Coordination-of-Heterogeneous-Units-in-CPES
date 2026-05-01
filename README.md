# Numerical Illustration for Flocking-Based Coordination of Heterogeneous Units in CPES

This repository contains the Python script accompanying the manuscript:

**A Formal Mapping Framework for Flocking-Based Coordination of Heterogeneous Units in Cyber-Physical Energy Systems**

The script provides a minimal numerical illustration of two theoretical mechanisms discussed in the paper:

1. separation acts as an anti-consensus component in the linearized position-to-acceleration map;
2. stress-gating removes this anti-consensus component under zero physical stress.

## Important scope limitation

This repository does **not** provide a validated CPES controller.

The numerical experiment is a qualitative mechanism illustration only. It uses one chain topology, one random seed, one parameter set, and a simplified algebraic weak-coupling model. It does not include measurement noise, communication delay, realistic power-flow simulation, or thermal plant modeling.

Robustness assessment requires Monte Carlo analysis, parameter sensitivity, multiple topologies, realistic plant models, measurement noise, communication delays, and comparison with baseline CPES controllers.

## Files

- `demo.py`: main Python script for the numerical illustration.
- `requirements.txt`: Python package requirements.
- `.gitignore`: ignored generated files and local Python artifacts.
- `LICENSE`: software license.

## Requirements

The script was prepared for Python 3.10 or newer.

Required package:

- `numpy`

Optional package for figure export:

- `matplotlib`

## Installation

Clone the repository:

```bash
git clone https://github.com/kovariati/Numerical-Illustration-for-Flocking-Based-Coordination-of-Heterogeneous-Units-in-CPES.git
cd Numerical-Illustration-for-Flocking-Based-Coordination-of-Heterogeneous-Units-in-CPES
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Basic run

To print the summary table:

```bash
python demo.py
```

## Export CSV traces and figures

To export reproducibility files:

```bash
python demo.py --export --output-dir demo_outputs
```

To export CSV files without PNG figures:

```bash
python demo.py --export --no-plots --output-dir demo_outputs
```

## Reported metrics

The script reports three aggregate metrics:

- `J_total_actuator_utilization`: total actuator utilization.
- `D_disagreement_energy`: disagreement energy.
- `stress_sum`: aggregate physical stress.

The disagreement energy `D` is the metric directly aligned with the anti-consensus interpretation. The actuator-utilization metric `J` is retained as a physical-activity proxy.

## Output files

With the `--export` option, the script creates:

- `scenario_summary.csv`: scenario-level summary table.
- `*_aggregate_trace.csv`: aggregate time traces for `J`, `D`, and total stress.
- `*_x_trace.csv`: unit-level behavioral-state traces.
- `*_v_trace.csv`: unit-level behavioral-velocity traces.
- `*_voltage_trace.csv`: unit-level measured-variable traces.
- `*_J_trace.png`, `*_D_trace.png`, and `*_stress_trace.png`: optional figures if `matplotlib` is installed.

## Reproducibility note

The script uses a fixed seed (`SEED = 42`) and the NumPy `RandomState` sequence to reproduce the numerical table reported in the manuscript.

## Citation

If this repository is used, please cite the accompanying manuscript after publication.

## License

This repository is distributed under the MIT License.
