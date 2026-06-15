from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from path_utils import resolve_config_path, resolve_input_path, resolve_output_path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


FEATURES = [
    "severity", "climate", "year", "day_of_year", "T_amb_C", "T_max_C",
    "RH_mean_pct", "GHI_mean_Wm2", "occ", "T_sp_C", "alpha_flow",
    "R_f", "dust_kg", "dP_Pa", "delta", "Q_cool_des_kw",
    "Q_heat_des_kw", "Q_air_nom_m3h", "area_m2"
]
CATEGORICAL = ["severity", "climate"]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]
TARGETS = ["energy_kwh_period", "comfort_dev_C", "maintenance_cost_usd"]
AUDIT_FIELDS = ["accepted_degradation_state_updates", "state_update_consistency_flag"]
DECISION_NAMES = ["T0", "k_delta_T", "k_occ_T", "alpha0", "k_delta_alpha", "k_occ_alpha"]


@dataclass
class Individual:
    x: np.ndarray
    f: np.ndarray | None = None
    rank: int = 0
    crowding: float = 0.0


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_and_filter(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    required = set(FEATURES + TARGETS + ["strategy", "scenario_combo_3axis", "time_scale_days"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    audit = {
        "row_count": int(len(df)),
        "has_corrected_state_audit_fields": all(c in df.columns for c in AUDIT_FIELDS),
        "missing_audit_fields": [c for c in AUDIT_FIELDS if c not in df.columns],
    }
    if cfg.get("require_corrected_audit_fields", False) and not audit["has_corrected_state_audit_fields"]:
        raise ValueError("Corrected state-update audit fields are required but absent.")
    if not audit["has_corrected_state_audit_fields"]:
        warnings.warn(
            "The dataset lacks corrected state-update audit fields. Results are exploratory and inherit any errors in the source dataset.",
            RuntimeWarning,
        )

    out = df[df["strategy"].astype(str) == str(cfg["strategy_for_surrogate"])].copy()
    excluded = set(map(str, cfg.get("exclude_severities", [])))
    out = out[~out["severity"].astype(str).isin(excluded)].copy()
    out = out.dropna(subset=FEATURES + TARGETS)
    if len(out) < 500:
        raise ValueError(f"Only {len(out)} usable rows remain after filtering; at least 500 are recommended.")
    audit["filtered_row_count"] = int(len(out))
    audit["severities"] = sorted(out["severity"].astype(str).unique().tolist())
    audit["control_support"] = {
        "T_sp_C_min": float(out["T_sp_C"].min()),
        "T_sp_C_max": float(out["T_sp_C"].max()),
        "alpha_flow_min": float(out["alpha_flow"].min()),
        "alpha_flow_max": float(out["alpha_flow"].max()),
    }
    return out, audit


def build_model(cfg: dict) -> Pipeline:
    pre = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("numeric", "passthrough", NUMERIC),
    ])
    model = ExtraTreesRegressor(
        n_estimators=int(cfg.get("trees", 250)),
        min_samples_leaf=int(cfg.get("min_samples_leaf", 2)),
        random_state=int(cfg.get("random_seed", 42)),
        n_jobs=-1,
        max_features=0.9,
    )
    return Pipeline([("preprocess", pre), ("model", model)])


def train_models(df: pd.DataFrame, cfg: dict, out: Path) -> tuple[dict[str, Pipeline], pd.DataFrame]:
    groups = df["scenario_combo_3axis"].astype(str) + "_Y" + df["year"].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=float(cfg.get("test_fraction", 0.2)), random_state=int(cfg.get("random_seed", 42)))
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    models: dict[str, Pipeline] = {}
    metrics = []
    model_dir = out / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    for target in TARGETS:
        pipe = build_model(cfg)
        pipe.fit(df.iloc[train_idx][FEATURES], df.iloc[train_idx][target])
        pred = pipe.predict(df.iloc[test_idx][FEATURES])
        y = df.iloc[test_idx][target].to_numpy(float)
        metrics.append({
            "target": target,
            "test_rows": int(len(test_idx)),
            "R2": float(r2_score(y, pred)),
            "MAE": float(mean_absolute_error(y, pred)),
            "RMSE": float(math.sqrt(mean_squared_error(y, pred))),
            "target_mean": float(np.mean(y)),
        })
        joblib.dump(pipe, model_dir / f"{target}.joblib", compress=3)
        models[target] = pipe

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(out / "surrogate_validation_metrics.csv", index=False)
    return models, metrics_df


def prepare_contexts(df: pd.DataFrame, cfg: dict) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    rng = np.random.default_rng(int(cfg.get("random_seed", 42)))
    n_req = int(cfg.get("context_rows_per_severity", 500))
    contexts = {}
    full_counts = {}
    for sev, g in df.groupby("severity", sort=True):
        g = g.sort_values(["year", "day_of_year"]).copy()
        full_counts[str(sev)] = int(len(g))
        if n_req > 0 and len(g) > n_req:
            idx = rng.choice(len(g), size=n_req, replace=False)
            sample = g.iloc[np.sort(idx)].copy()
        else:
            sample = g.copy()
        contexts[str(sev)] = sample
    return contexts, full_counts


def make_bounds(cfg: dict, support: dict) -> tuple[np.ndarray, np.ndarray]:
    pb = cfg["policy_bounds"]
    lo = np.array([pb[n][0] for n in DECISION_NAMES], dtype=float)
    hi = np.array([pb[n][1] for n in DECISION_NAMES], dtype=float)
    # The policy output itself is clipped to observed support, avoiding direct control extrapolation.
    if lo.shape != (6,) or hi.shape != (6,):
        raise ValueError("policy_bounds must contain all six decision names.")
    return lo, hi


def policy_controls(x: np.ndarray, context: pd.DataFrame, support: dict, delta_ref: float) -> tuple[np.ndarray, np.ndarray]:
    T0, kdT, koT, a0, kda, koa = map(float, x)
    delta_c = context["delta"].to_numpy(float) - delta_ref
    occ = context["occ"].to_numpy(float)
    tsp = T0 + kdT * delta_c + koT * (1.0 - occ)
    alpha = a0 + kda * delta_c + koa * occ
    tsp = np.clip(tsp, support["T_sp_C_min"], support["T_sp_C_max"])
    alpha = np.clip(alpha, support["alpha_flow_min"], support["alpha_flow_max"])
    return tsp, alpha


def make_evaluator(models: dict[str, Pipeline], contexts: dict[str, pd.DataFrame], full_counts: dict[str, int], support: dict, cfg: dict) -> Callable[[np.ndarray], tuple[np.ndarray, dict]]:
    delta_ref = float(np.mean(np.concatenate([g["delta"].to_numpy(float) for g in contexts.values()])))

    def evaluate(x: np.ndarray) -> tuple[np.ndarray, dict]:
        per_severity = []
        for sev, g in contexts.items():
            work = g[FEATURES].copy()
            tsp, alpha = policy_controls(x, g, support, delta_ref)
            work["T_sp_C"] = tsp
            work["alpha_flow"] = alpha
            e = np.maximum(models["energy_kwh_period"].predict(work), 0.0)
            c = np.maximum(models["comfort_dev_C"].predict(work), 0.0)
            m = np.maximum(models["maintenance_cost_usd"].predict(work), 0.0)
            scale = full_counts[sev] / len(g)
            per_severity.append({
                "severity": sev,
                "energy_MWh": float(e.sum() * scale / 1000.0),
                "occupied_comfort_degree_days": float((c * g["occ"].to_numpy(float) * g["time_scale_days"].to_numpy(float)).sum() * scale),
                "direct_maintenance_cost_USD": float(m.sum() * scale),
                "mean_T_sp_C": float(tsp.mean()),
                "mean_alpha_flow": float(alpha.mean()),
                "T_sp_at_lower_support_fraction": float(np.mean(np.isclose(tsp, support["T_sp_C_min"]))),
                "T_sp_at_upper_support_fraction": float(np.mean(np.isclose(tsp, support["T_sp_C_max"]))),
                "alpha_at_lower_support_fraction": float(np.mean(np.isclose(alpha, support["alpha_flow_min"]))),
                "alpha_at_upper_support_fraction": float(np.mean(np.isclose(alpha, support["alpha_flow_max"]))),
            })
        d = pd.DataFrame(per_severity)
        F = np.array([
            d["energy_MWh"].mean(),
            d["occupied_comfort_degree_days"].max(),
            d["direct_maintenance_cost_USD"].mean(),
        ], dtype=float)
        details = {
            "per_severity": d,
            "mean_T_sp_C": float(d["mean_T_sp_C"].mean()),
            "mean_alpha_flow": float(d["mean_alpha_flow"].mean()),
            "max_boundary_fraction": float(d[[c for c in d.columns if c.endswith("support_fraction")]].to_numpy().max()),
        }
        return F, details

    return evaluate


# --------------------- Minimal offline NSGA-II ---------------------
def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def fast_nondominated_sort(pop: list[Individual]) -> list[list[int]]:
    S = [[] for _ in pop]
    n = np.zeros(len(pop), dtype=int)
    fronts: list[list[int]] = [[]]
    for p in range(len(pop)):
        for q in range(len(pop)):
            if p == q:
                continue
            if dominates(pop[p].f, pop[q].f):
                S[p].append(q)
            elif dominates(pop[q].f, pop[p].f):
                n[p] += 1
        if n[p] == 0:
            pop[p].rank = 0
            fronts[0].append(p)
    i = 0
    while i < len(fronts) and fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    pop[q].rank = i + 1
                    nxt.append(q)
        i += 1
        if nxt:
            fronts.append(nxt)
    return fronts


def crowding_distance(pop: list[Individual], front: list[int]) -> None:
    if not front:
        return
    for i in front:
        pop[i].crowding = 0.0
    m = len(pop[front[0]].f)
    for obj in range(m):
        ordered = sorted(front, key=lambda i: pop[i].f[obj])
        pop[ordered[0]].crowding = pop[ordered[-1]].crowding = float("inf")
        fmin, fmax = pop[ordered[0]].f[obj], pop[ordered[-1]].f[obj]
        if abs(fmax - fmin) < 1e-15:
            continue
        for k in range(1, len(ordered)-1):
            pop[ordered[k]].crowding += (pop[ordered[k+1]].f[obj] - pop[ordered[k-1]].f[obj]) / (fmax - fmin)


def rank_population(pop: list[Individual]) -> list[list[int]]:
    fronts = fast_nondominated_sort(pop)
    for front in fronts:
        crowding_distance(pop, front)
    return fronts


def tournament(rng: np.random.Generator, pop: list[Individual]) -> Individual:
    a, b = rng.integers(0, len(pop), size=2)
    ia, ib = pop[a], pop[b]
    if ia.rank < ib.rank:
        return ia
    if ib.rank < ia.rank:
        return ib
    return ia if ia.crowding >= ib.crowding else ib


def sbx(rng: np.random.Generator, p1: np.ndarray, p2: np.ndarray, lo: np.ndarray, hi: np.ndarray, eta: float, prob: float) -> tuple[np.ndarray, np.ndarray]:
    c1, c2 = p1.copy(), p2.copy()
    if rng.random() > prob:
        return c1, c2
    for i in range(len(p1)):
        if rng.random() > 0.5 or abs(p1[i]-p2[i]) < 1e-14:
            continue
        x1, x2 = sorted([p1[i], p2[i]])
        rand = rng.random()
        beta = 1.0 + 2.0*(x1-lo[i])/(x2-x1)
        alpha = 2.0 - beta**(-(eta+1.0))
        betaq = (rand*alpha)**(1.0/(eta+1.0)) if rand <= 1.0/alpha else (1.0/(2.0-rand*alpha))**(1.0/(eta+1.0))
        child1 = 0.5*((x1+x2)-betaq*(x2-x1))
        beta = 1.0 + 2.0*(hi[i]-x2)/(x2-x1)
        alpha = 2.0 - beta**(-(eta+1.0))
        betaq = (rand*alpha)**(1.0/(eta+1.0)) if rand <= 1.0/alpha else (1.0/(2.0-rand*alpha))**(1.0/(eta+1.0))
        child2 = 0.5*((x1+x2)+betaq*(x2-x1))
        if rng.random() <= 0.5:
            c1[i], c2[i] = child2, child1
        else:
            c1[i], c2[i] = child1, child2
    return np.clip(c1, lo, hi), np.clip(c2, lo, hi)


def polynomial_mutation(rng: np.random.Generator, x: np.ndarray, lo: np.ndarray, hi: np.ndarray, eta: float, prob: float) -> np.ndarray:
    y = x.copy()
    for i in range(len(y)):
        if rng.random() > prob:
            continue
        delta1 = (y[i]-lo[i])/(hi[i]-lo[i])
        delta2 = (hi[i]-y[i])/(hi[i]-lo[i])
        r = rng.random()
        mut_pow = 1.0/(eta+1.0)
        if r < 0.5:
            xy = 1.0-delta1
            val = 2.0*r + (1.0-2.0*r)*(xy**(eta+1.0))
            deltaq = val**mut_pow - 1.0
        else:
            xy = 1.0-delta2
            val = 2.0*(1.0-r) + 2.0*(r-0.5)*(xy**(eta+1.0))
            deltaq = 1.0 - val**mut_pow
        y[i] = np.clip(y[i] + deltaq*(hi[i]-lo[i]), lo[i], hi[i])
    return y


def environmental_selection(combined: list[Individual], n: int) -> list[Individual]:
    fronts = rank_population(combined)
    new: list[Individual] = []
    for front in fronts:
        if len(new) + len(front) <= n:
            new.extend(combined[i] for i in front)
        else:
            remaining = n-len(new)
            chosen = sorted(front, key=lambda i: combined[i].crowding, reverse=True)[:remaining]
            new.extend(combined[i] for i in chosen)
            break
    rank_population(new)
    return new


def run_nsga2(evaluate: Callable[[np.ndarray], tuple[np.ndarray, dict]], lo: np.ndarray, hi: np.ndarray, cfg: dict) -> tuple[list[Individual], pd.DataFrame]:
    rng = np.random.default_rng(int(cfg.get("random_seed", 42)))
    n_pop = int(cfg.get("population_size", 32))
    n_gen = int(cfg.get("generations", 30))
    mut_prob = float(cfg.get("mutation_probability", 1.0/len(lo)))
    cross_prob = float(cfg.get("crossover_probability", 0.9))
    sbx_eta = float(cfg.get("sbx_eta", 15.0))
    mut_eta = float(cfg.get("mutation_eta", 20.0))

    pop = [Individual(x=rng.uniform(lo, hi)) for _ in range(n_pop)]
    for ind in pop:
        ind.f, _ = evaluate(ind.x)
    rank_population(pop)
    history = []

    for gen in range(1, n_gen+1):
        children: list[Individual] = []
        while len(children) < n_pop:
            p1, p2 = tournament(rng, pop), tournament(rng, pop)
            c1, c2 = sbx(rng, p1.x, p2.x, lo, hi, sbx_eta, cross_prob)
            c1 = polynomial_mutation(rng, c1, lo, hi, mut_eta, mut_prob)
            c2 = polynomial_mutation(rng, c2, lo, hi, mut_eta, mut_prob)
            children.append(Individual(x=c1))
            if len(children) < n_pop:
                children.append(Individual(x=c2))
        for ind in children:
            ind.f, _ = evaluate(ind.x)
        pop = environmental_selection(pop + children, n_pop)
        front0 = [ind for ind in pop if ind.rank == 0]
        F0 = np.vstack([i.f for i in front0])
        history.append({
            "generation": gen,
            "nondominated_solutions": len(front0),
            "minimum_energy_MWh": float(F0[:,0].min()),
            "minimum_comfort_degree_days": float(F0[:,1].min()),
            "minimum_maintenance_cost_USD": float(F0[:,2].min()),
        })
        print(f"Generation {gen}/{n_gen}: nondominated={len(front0)}")
    return pop, pd.DataFrame(history)


def select_closest_to_ideal(F: np.ndarray) -> tuple[int, np.ndarray]:
    lo, hi = F.min(axis=0), F.max(axis=0)
    norm = (F-lo)/(hi-lo+1e-12)
    dist = np.linalg.norm(norm, axis=1)
    return int(np.argmin(dist)), dist


def run_surrogate_analysis(
    input_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    config_overrides: dict | None = None,
) -> dict:
    source = resolve_input_path(input_path)
    config_source = resolve_config_path(config_path)
    cfg = load_json(config_source)
    if config_overrides:
        cfg.update(config_overrides)
    out = resolve_output_path(output_dir, "outputs/surrogate_nsga2")

    raw = pd.read_csv(source)
    df, audit = validate_and_filter(raw, cfg)
    with open(out/"dataset_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    models, metrics = train_models(df, cfg, out)
    contexts, full_counts = prepare_contexts(df, cfg)
    evaluate = make_evaluator(models, contexts, full_counts, audit["control_support"], cfg)
    lo, hi = make_bounds(cfg, audit["control_support"])
    pop, history = run_nsga2(evaluate, lo, hi, cfg)
    history.to_csv(out/"convergence_history.csv", index=False)

    pareto = [ind for ind in pop if ind.rank == 0]
    X = np.vstack([i.x for i in pareto])
    F = np.vstack([i.f for i in pareto])
    selected_idx, distances = select_closest_to_ideal(F)

    rows = []
    for i, ind in enumerate(pareto):
        _, details = evaluate(ind.x)
        row = {DECISION_NAMES[j]: float(ind.x[j]) for j in range(6)}
        row.update({
            "mean_energy_MWh": float(ind.f[0]),
            "worst_occupied_comfort_degree_days": float(ind.f[1]),
            "mean_direct_maintenance_cost_USD": float(ind.f[2]),
            "mean_T_sp_C": details["mean_T_sp_C"],
            "mean_alpha_flow": details["mean_alpha_flow"],
            "max_boundary_fraction": details["max_boundary_fraction"],
            "distance_to_ideal": float(distances[i]),
            "selected_closest_to_ideal": bool(i == selected_idx),
        })
        rows.append(row)
    pareto_df = pd.DataFrame(rows).sort_values("distance_to_ideal").reset_index(drop=True)
    pareto_df.to_csv(out/"pareto_policy_solutions.csv", index=False)

    selected_x = X[selected_idx]
    selected_f, selected_details = evaluate(selected_x)
    selected_details["per_severity"].to_csv(out/"selected_policy_by_severity.csv", index=False)
    pd.DataFrame([{**{DECISION_NAMES[j]: float(selected_x[j]) for j in range(6)},
                       "mean_energy_MWh": float(selected_f[0]),
                       "worst_occupied_comfort_degree_days": float(selected_f[1]),
                       "mean_direct_maintenance_cost_USD": float(selected_f[2])}]).to_csv(out/"selected_policy.csv", index=False)

    plt.figure(figsize=(8,6))
    sc = plt.scatter(F[:,0], F[:,1], c=F[:,2], s=60)
    plt.scatter(F[selected_idx,0], F[selected_idx,1], marker="*", s=260, label="Closest-to-ideal")
    plt.xlabel("Predicted mean five-year energy (MWh)")
    plt.ylabel("Predicted worst occupied comfort degree-days")
    plt.colorbar(sc, label="Predicted mean direct maintenance cost (USD)")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out/"pareto_energy_comfort.png", dpi=400)
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(history["generation"], history["nondominated_solutions"], marker="o")
    plt.xlabel("Generation")
    plt.ylabel("Nondominated solutions")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out/"convergence_nondominated_count.png", dpi=400)
    plt.close()

    selected_policy = pd.read_csv(out / "selected_policy.csv")
    return {
        "input_path": source,
        "output_dir": out,
        "audit": audit,
        "metrics": metrics,
        "pareto": pareto_df,
        "selected_policy": selected_policy,
        "selected_by_severity": selected_details["per_severity"],
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline surrogate-assisted NSGA-II using core-solver output CSV data.")
    parser.add_argument("--input", default=None, help="CSV path. Defaults to bundle/data/matrix_ml_dataset.csv.")
    parser.add_argument("--config", default=None, help="Configuration JSON. Defaults to bundle/config.json.")
    parser.add_argument("--output", default=None, help="Output directory. Defaults inside this bundle.")
    args = parser.parse_args()

    result = run_surrogate_analysis(args.input, args.config, args.output)
    print("\nCompleted.")
    print(f"Input: {result['input_path']}")
    print(f"Validation metrics:\n{result['metrics'].to_string(index=False)}")
    print(f"Selected policy:\n{result['selected_policy'].to_string(index=False)}")
    print(f"Outputs saved to: {result['output_dir']}")


if __name__ == "__main__":
    main()
