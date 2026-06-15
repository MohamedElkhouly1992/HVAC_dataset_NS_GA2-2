from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from path_utils import resolve_input_path, resolve_output_path


REQUIRED_COLUMNS = [
    "strategy", "severity", "energy_kwh_period", "comfort_dev_C", "occ",
    "time_scale_days", "maintenance_cost_usd", "filter_replaced", "hx_cleaned",
    "co2_kg_period", "delta",
]


def nondominated_mask(F: np.ndarray) -> np.ndarray:
    n = len(F)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        dominates_i = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        dominates_i[i] = False
        if np.any(dominates_i):
            keep[i] = False
    return keep


def analyze_observed_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = df.copy()
    # Older baseline rows can contain NaN maintenance cost. They represent no action.
    for column in ["maintenance_cost_usd", "filter_replaced", "hx_cleaned"]:
        work[column] = pd.to_numeric(work[column], errors="coerce").fillna(0.0)

    grouped = work.groupby(["strategy", "severity"], as_index=False).agg(
        total_energy_MWh=("energy_kwh_period", lambda s: float(s.sum() / 1000.0)),
        direct_maintenance_cost_USD=("maintenance_cost_usd", "sum"),
        filter_events=("filter_replaced", "sum"),
        hx_events=("hx_cleaned", "sum"),
        total_CO2_tonne=("co2_kg_period", lambda s: float(s.sum() / 1000.0)),
        mean_degradation=("delta", "mean"),
    )

    comfort = (
        pd.to_numeric(work["comfort_dev_C"], errors="coerce").fillna(0.0)
        * pd.to_numeric(work["occ"], errors="coerce").fillna(0.0)
        * pd.to_numeric(work["time_scale_days"], errors="coerce").fillna(0.0)
    )
    comfort_by_group = (
        work.assign(_comfort_degree_days=comfort)
        .groupby(["strategy", "severity"], as_index=False)["_comfort_degree_days"]
        .sum()
        .rename(columns={"_comfort_degree_days": "occupied_comfort_degree_days"})
    )
    grouped = grouped.merge(comfort_by_group, on=["strategy", "severity"], how="left")

    objective_columns = [
        "total_energy_MWh",
        "occupied_comfort_degree_days",
        "direct_maintenance_cost_USD",
    ]
    F = grouped[objective_columns].to_numpy(float)
    grouped["pareto_nondominated"] = nondominated_mask(F)

    pareto = grouped[grouped["pareto_nondominated"]].copy()
    pareto["distance_to_ideal"] = np.nan
    pareto["selected_closest_to_ideal"] = False
    if not pareto.empty:
        pF = pareto[objective_columns].to_numpy(float)
        lower = pF.min(axis=0)
        upper = pF.max(axis=0)
        normalized = (pF - lower) / (upper - lower + 1e-12)
        pareto["distance_to_ideal"] = np.linalg.norm(normalized, axis=1)
        chosen_label = pareto["distance_to_ideal"].idxmin()
        pareto.loc[chosen_label, "selected_closest_to_ideal"] = True

    return grouped, pareto


def run_observed_analysis(input_path: str | Path, output_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = resolve_input_path(input_path)
    output = resolve_output_path(output_dir, "outputs/observed")
    df = pd.read_csv(source)
    grouped, pareto = analyze_observed_dataframe(df)
    grouped.to_csv(output / "observed_scenario_summary.csv", index=False)
    pareto.to_csv(output / "observed_pareto_solutions.csv", index=False)
    return grouped, pareto


def main() -> None:
    parser = argparse.ArgumentParser(description="Observed Pareto analysis from core-solver CSV outputs.")
    parser.add_argument("--input", default=None, help="CSV path. Defaults to bundle/data/matrix_ml_dataset.csv.")
    parser.add_argument("--output", default=None, help="Output directory. Defaults inside this bundle.")
    args = parser.parse_args()

    source = resolve_input_path(args.input)
    output = resolve_output_path(args.output, "outputs/observed")
    grouped, pareto = run_observed_analysis(source, output)
    print(f"Input: {source}")
    print(f"Scenarios: {len(grouped)}; Pareto solutions: {len(pareto)}")
    print(f"Outputs saved to: {output}")


if __name__ == "__main__":
    main()
