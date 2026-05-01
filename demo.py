"""
Numerical illustration for paper titled:
"A Formal Mapping Framework for Flocking-Based Coordination
 of Heterogeneous Units in Cyber-Physical Energy Systems"

PURPOSE
-------
This script illustrates two theoretical claims:
  Theorem 2  - separation acts as an anti-consensus component in the
               linearized position-to-acceleration map.
  Corollary 1 - stress-gating removes this anti-consensus component
                under zero physical stress.

The Reynolds-style drives are adapted to a one-dimensional normalized
CPES behavioral coordinate. They are inspired by the OpenSteer separation,
alignment, and cohesion logic, but the separation kernel used here is a
regularized scalar analogue rather than a literal multi-dimensional
inverse-square boids implementation.

SYNCHRONIZATION
---------------
All drives use (x_old, v_old) from the start of each time step
(Jacobi/synchronous update), matching the paper definition.

LIMITATION
----------
Single topology, single seed, single parameter set, algebraic weak-coupling
model. This is not a validation. Robustness assessment requires Monte Carlo
analysis, parameter sensitivity, multiple topologies, noise, delays, and
realistic physical models.

OUTPUTS
-------
By default, the script prints the summary table used in the manuscript.
With --export, it also writes CSV traces and optional PNG figures to the
chosen output directory. The reported metrics distinguish total actuator
utilization J from disagreement energy D.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np


# ── Parameters ──────────────────────────────────────────────────
N = 10
Y_REF = 1.00
DELTA = 0.01
STRESS_TOL = 0.001
G_DROOP = 80.0
U_MAX = np.array([3.0 + 0.5 * i for i in range(N)], dtype=float)

KAPPA = 0.20
EPS_S = 0.10
F_MAX = 0.15
V_MAX = 0.12
MASS = 1.0
DT = 1.0
LAMBDA_R = 0.5
LAMBDA_T = 1.0
W_S = 0.5
W_A = 0.3
W_C = 0.2
BETA_REST = 0.2

SEED = 42
VALID_MODES = {"standard", "gated"}


# Weak algebraic coupling used only for illustration.
Z = 0.0005
S = np.full((N, N), Z * 0.1, dtype=float)
for i in range(N):
    for j in range(N):
        if abs(i - j) <= 2:
            S[i, j] = Z


# Graph-hop neighborhoods; each excludes the focal unit.
NS = {i: [j for j in range(N) if j != i and abs(i - j) <= 1] for i in range(N)}
NA = {i: [j for j in range(N) if j != i and abs(i - j) <= 2] for i in range(N)}
NC = {i: [j for j in range(N) if j != i and abs(i - j) <= 4] for i in range(N)}


def soft_norm(z: float) -> float:
    """Soft-normalization N(z)=z/(|z|+kappa)."""
    return z / (abs(z) + KAPPA)


def stress(y: float) -> float:
    """Symmetric deadband stress."""
    return max(0.0, abs(y - Y_REF) - DELTA)


def disagreement_energy(x: np.ndarray) -> float:
    """Return D=sum_i (x_i - mean(x))^2.

    J=sum_i |x_i| measures total actuator utilization. D measures
    inter-unit disagreement in the normalized behavioral coordinate.
    """
    x_mean = float(np.mean(x))
    return float(np.sum((x - x_mean) ** 2))


def disturbance_vector(kind: str) -> Optional[np.ndarray]:
    """Return the disturbance vector for a named scenario."""
    if kind == "zero":
        return None
    if kind == "full":
        return np.array([0.02 * (i % 5 + 1) for i in range(N)], dtype=float)
    if kind == "partial":
        d = np.zeros(N, dtype=float)
        d[5:] = np.array([0.03 * (i - 4) for i in range(5, N)], dtype=float)
        return d
    raise ValueError(f"unknown disturbance kind: {kind}")


def run(
    mode: str,
    steps: int,
    perturb: bool,
    disturbance: Optional[np.ndarray] = None,
    seed: int = SEED,
) -> Dict[str, np.ndarray]:
    """Run one illustrative simulation.

    Parameters
    ----------
    mode:
        "standard" keeps separation always active. "gated" applies the
        stress-gated separation neighborhood.
    steps:
        Number of discrete-time steps.
    perturb:
        If True, use a small random initial perturbation in x.
    disturbance:
        Optional constant disturbance vector d.
    seed:
        Random seed used only when perturb=True.

    Returns
    -------
    Dictionary containing J, disagreement energy D, stress_sum, x_trace, v_trace and voltage_trace.
    Traces are recorded after each simultaneous state update.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_MODES)}, got {mode!r}")

    # Use the legacy RandomState sequence to reproduce the manuscript table exactly.
    rng = np.random.RandomState(seed)
    x = rng.normal(0.0, 0.01, size=N) if perturb else np.zeros(N, dtype=float)
    v = np.zeros(N, dtype=float)
    d = np.zeros(N, dtype=float) if disturbance is None else disturbance.astype(float)

    j_trace = []
    d_trace = []
    stress_trace = []
    x_trace = []
    v_trace = []
    voltage_trace = []

    for _ in range(steps):
        V = Y_REF + S @ (U_MAX * x) + d
        stress_now = np.array([stress(V[i]) for i in range(N)])

        # Jacobi: snapshot state before computing all drives.
        x_old = x.copy()
        v_old = v.copy()
        x_new = np.zeros(N, dtype=float)
        v_new = np.zeros(N, dtype=float)

        for i in range(N):
            sep_neighbors = list(NS[i])
            if mode == "gated":
                sep_neighbors = [
                    j for j in sep_neighbors
                    if stress_now[i] >= STRESS_TOL or stress_now[j] >= STRESS_TOL
                ]

            if sep_neighbors:
                sep_sum = np.mean([
                    (x_old[i] - x_old[j]) / ((x_old[i] - x_old[j]) ** 2 + EPS_S ** 2)
                    for j in sep_neighbors
                ])
                d_sep = soft_norm(float(sep_sum))
            else:
                d_sep = 0.0

            d_align = (
                soft_norm(float(np.mean([v_old[j] for j in NA[i]]) - v_old[i]))
                if NA[i] else 0.0
            )
            d_cohesion = (
                soft_norm(float(np.mean([x_old[j] for j in NC[i]]) - x_old[i]))
                if NC[i] else 0.0
            )

            # Local droop target. Positive voltage deviation produces negative Q
            # under the stated sign convention.
            r = -np.sign(V[i] - Y_REF) * max(0.0, abs(V[i] - Y_REF) - DELTA) * G_DROOP
            d_target = float(np.clip(r / U_MAX[i] - x_old[i], -1.0, 1.0))

            a = np.clip(
                LAMBDA_R * (W_S * d_sep + W_A * d_align + W_C * d_cohesion)
                + LAMBDA_T * d_target,
                -F_MAX,
                F_MAX,
            )
            vi = np.clip(v_old[i] + DT / MASS * a, -V_MAX, V_MAX)

            # Rest-state decay uses the widest cohesion neighborhood.
            if mode == "gated" and stress_now[i] < STRESS_TOL:
                if all(stress_now[j] < STRESS_TOL for j in NC[i]):
                    vi *= BETA_REST

            x_new[i] = np.clip(x_old[i] + DT * vi, -1.0, 1.0)
            v_new[i] = vi

        # Simultaneous update.
        x[:] = x_new
        v[:] = v_new

        # Record after update, avoiding an off-by-one interpretation.
        V_post = Y_REF + S @ (U_MAX * x) + d
        stress_post = np.array([stress(V_post[i]) for i in range(N)])

        j_trace.append(float(np.sum(np.abs(x))))
        d_trace.append(disagreement_energy(x))
        stress_trace.append(float(np.sum(stress_post)))
        x_trace.append(x.copy())
        v_trace.append(v.copy())
        voltage_trace.append(V_post.copy())

    return {
        "J": np.array(j_trace),
        "D": np.array(d_trace),
        "stress": np.array(stress_trace),
        "x_final": x.copy(),
        "v_final": v.copy(),
        "x_trace": np.array(x_trace),
        "v_trace": np.array(v_trace),
        "voltage_trace": np.array(voltage_trace),
    }


def scenario_specs():
    """Scenario definitions used in the manuscript."""
    return [
        {
            "name": "scenario1_zero_disturbance",
            "label": "Scenario 1: Zero disturbance, small perturbation",
            "steps": 300,
            "perturb": True,
            "disturbance": disturbance_vector("zero"),
        },
        {
            "name": "scenario2_full_stress",
            "label": "Scenario 2: Full stress (all units stressed)",
            "steps": 100,
            "perturb": False,
            "disturbance": disturbance_vector("full"),
        },
        {
            "name": "scenario3_partial_stress",
            "label": "Scenario 3: Partial stress (units 5-9 only)",
            "steps": 200,
            "perturb": False,
            "disturbance": disturbance_vector("partial"),
        },
    ]


def summarize_result(scenario_name: str, mode: str, result: Dict[str, np.ndarray]) -> Dict[str, float | str]:
    """Create one summary-row dictionary."""
    return {
        "scenario": scenario_name,
        "mode": mode,
        "J_after_update_1": float(result["J"][0]),
        "J_final": float(result["J"][-1]),
        "D_after_update_1": float(result["D"][0]),
        "D_final": float(result["D"][-1]),
        "stress_final": float(result["stress"][-1]),
        "trend": "D_AMPLIFIED" if result["D"][-1] > result["D"][0] + 1e-6 else "D_DAMPED_OR_STABLE",
    }


def write_summary_csv(rows: Iterable[Dict[str, float | str]], path: Path) -> None:
    """Write scenario-level summary CSV."""
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_trace_csv(scenario: str, mode: str, result: Dict[str, np.ndarray], output_dir: Path) -> None:
    """Write aggregate and unit-level traces for one scenario/mode pair."""
    aggregate_path = output_dir / f"{scenario}_{mode}_aggregate_trace.csv"
    with aggregate_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "J_total_actuator_utilization", "D_disagreement_energy", "stress_sum"])
        for step, (j_val, d_val, stress_val) in enumerate(zip(result["J"], result["D"], result["stress"]), start=1):
            writer.writerow([step, f"{j_val:.10f}", f"{d_val:.10f}", f"{stress_val:.10f}"])

    for key, prefix in [("x_trace", "x"), ("v_trace", "v"), ("voltage_trace", "voltage")]:
        trace_path = output_dir / f"{scenario}_{mode}_{prefix}_trace.csv"
        with trace_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step"] + [f"{prefix}_{i}" for i in range(N)])
            for step, row in enumerate(result[key], start=1):
                writer.writerow([step] + [f"{value:.10f}" for value in row])


def maybe_write_plots(all_results: Dict[str, Dict[str, Dict[str, np.ndarray]]], output_dir: Path) -> None:
    """Write simple line plots if matplotlib is installed."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"matplotlib not available; skipping figures ({exc})")
        return

    for scenario_name, modes in all_results.items():
        # J(t)
        plt.figure()
        for mode, result in modes.items():
            plt.plot(np.arange(1, len(result["J"]) + 1), result["J"], label=mode)
        plt.xlabel("Step")
        plt.ylabel("J = sum |x_i|")
        plt.title(scenario_name)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{scenario_name}_J_trace.png", dpi=200)
        plt.close()

        # D(t): disagreement energy
        plt.figure()
        for mode, result in modes.items():
            plt.plot(np.arange(1, len(result["D"]) + 1), result["D"], label=mode)
        plt.xlabel("Step")
        plt.ylabel("D = sum (x_i - mean(x))^2")
        plt.title(scenario_name)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{scenario_name}_D_trace.png", dpi=200)
        plt.close()

        # Stress(t)
        plt.figure()
        for mode, result in modes.items():
            plt.plot(np.arange(1, len(result["stress"]) + 1), result["stress"], label=mode)
        plt.xlabel("Step")
        plt.ylabel("Total stress")
        plt.title(scenario_name)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{scenario_name}_stress_trace.png", dpi=200)
        plt.close()


def print_summary_table(rows: Iterable[Dict[str, float | str]]) -> None:
    """Print the manuscript-level summary table."""
    rows = list(rows)
    scenario_names = []
    for row in rows:
        if row["scenario"] not in scenario_names:
            scenario_names.append(str(row["scenario"]))

    for scenario in scenario_names:
        print(f"\n--- {scenario} ---")
        print(f"{'Mode':30s} {'J(upd.1)':>12s} {'J(final)':>12s} {'D(upd.1)':>12s} {'D(final)':>12s} {'Stress_sum':>12s} {'Trend':>18s}")
        print("-" * 92)
        for row in [r for r in rows if r["scenario"] == scenario]:
            print(
                f"  {row['mode']:28s} "
                f"{row['J_after_update_1']:12.4f} "
                f"{row['J_final']:12.4f} "
                f"{row['D_after_update_1']:12.6f} "
                f"{row['D_final']:12.6f} "
                f"{row['stress_final']:12.4f} "
                f"{row['trend']:>18s}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the minimal CPES flocking illustration.")
    parser.add_argument("--export", action="store_true", help="export CSV traces and optional figures")
    parser.add_argument("--output-dir", type=Path, default=Path("demo_outputs"), help="output directory for exported files")
    parser.add_argument("--no-plots", action="store_true", help="skip optional PNG figure generation")
    args = parser.parse_args()

    print("=" * 78)
    print("Minimal Numerical Illustration (Jacobi/synchronous update)")
    print("10 units, chain graph, algebraic weak coupling, seed=42")
    print("This is an ILLUSTRATION, not a validation.")
    print("=" * 78)

    all_results: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    summary_rows = []

    for spec in scenario_specs():
        scenario_results = {}
        for mode in ["standard", "gated"]:
            result = run(
                mode=mode,
                steps=spec["steps"],
                perturb=spec["perturb"],
                disturbance=spec["disturbance"],
                seed=SEED,
            )
            scenario_results[mode] = result
            summary_rows.append(summarize_result(spec["name"], mode, result))
        all_results[spec["name"]] = scenario_results

    print_summary_table(summary_rows)

    if args.export:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_summary_csv(summary_rows, args.output_dir / "scenario_summary.csv")
        for scenario_name, modes in all_results.items():
            for mode, result in modes.items():
                write_trace_csv(scenario_name, mode, result, args.output_dir)
        if not args.no_plots:
            maybe_write_plots(all_results, args.output_dir)
        print(f"\nExported reproducibility files to: {args.output_dir.resolve()}")

    print("=" * 78)
    print("DISCLAIMER: This is an illustrative demonstration on a single topology,")
    print("single seed, single parameter set, and weak algebraic coupling model.")
    print("=" * 78)


if __name__ == "__main__":
    main()
