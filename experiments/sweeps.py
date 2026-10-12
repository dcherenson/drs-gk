"""DRS-GK experiment sweeps and result summarisation.

The public functions in this module are kept separate from the comparison
runner so that a sweep can be resumed from its CSV output or invoked from a
small, reproducible driver script.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from config import (
    EPS_BETA_SWEEP_CONFIG,
    GATEKEEPER_VISUAL_RADIUS,
    HYPERPARAM_SWEEP_CONFIG,
    INITIAL_STATE_COMPARISON_CONFIG,
    OBSTACLES,
    RUN_CONFIG,
    TRUE_BETA,
    gk_params,
    monte_carlo_params,
    state_0,
    pcis_params as DEFAULT_PCIS_PARAMS,
)
from controller.drs_gatekeeper import get_drs_gk_bounds, run_gatekeeper_simulation
from helper import collides_with_any_obstacle

from .monte_carlo_analysis import run_true_probability_monte_carlo


__all__ = [
    "run_one_at_a_time_hyperparameter_sweep",
    "run_epsilon_beta_sweep",
    "summarize_sweep_results",
    "save_summary_csv",
]


def _build_pcis_data(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
    """Load PCIS construction lazily so analysis helpers remain importable."""

    from pcis import build_pcis_data

    return build_pcis_data(*args, **kwargs)


def sample_initial_states_in_box(
    n_samples: int,
    seed: int | None = None,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    base_state: Any | None = None,
    obstacles: Any | None = None,
    avoid_obstacles: bool | None = None,
) -> list[jnp.ndarray]:
    """Sample initial states while preserving the nominal heading and speed."""

    if base_state is None:
        base_state = state_0
    if obstacles is None:
        obstacles = OBSTACLES
    if seed is None:
        seed = int(EPS_BETA_SWEEP_CONFIG["seed_start"])
    if x_range is None:
        x_range = tuple(EPS_BETA_SWEEP_CONFIG["x_range"])
    if y_range is None:
        y_range = tuple(EPS_BETA_SWEEP_CONFIG["y_range"])
    if avoid_obstacles is None:
        avoid_obstacles = bool(INITIAL_STATE_COMPARISON_CONFIG.get("avoid_obstacles", False))

    base_state = jnp.asarray(base_state)
    rng = np.random.default_rng(int(seed))
    sampled_states = []
    while len(sampled_states) < int(n_samples):
        px = float(rng.uniform(*x_range))
        py = float(rng.uniform(*y_range))
        candidate = jnp.array([px, py, float(base_state[2]), float(base_state[3])])
        if avoid_obstacles and collides_with_any_obstacle(candidate, obstacles):
            continue
        sampled_states.append(candidate)
    return sampled_states


def build_sweep_values(spec: Iterable[Any]) -> list[float]:
    """Convert a configured sweep vector to plain Python numeric values."""

    return np.asarray(spec, dtype=float).tolist()


def values_match(a: Any, b: Any) -> bool:
    """Compare integer and floating sweep values without string round trips."""

    if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
        return int(a) == int(b)
    return bool(np.isclose(float(a), float(b), rtol=1e-9, atol=1e-12))


def build_one_at_a_time_trials(
    nominal_params: Mapping[str, Any], sweep_specs: Mapping[str, Iterable[Any]]
) -> list[dict[str, Any]]:
    """Build nominal plus one-parameter-at-a-time trial configurations."""

    trials = [
        {
            "swept_param": "nominal",
            "swept_value": np.nan,
            "params": dict(nominal_params),
        }
    ]
    for param_name, spec in sweep_specs.items():
        for swept_value in build_sweep_values(spec):
            if values_match(swept_value, nominal_params[param_name]):
                continue
            trial_params = dict(nominal_params)
            nominal_value = nominal_params[param_name]
            if isinstance(nominal_value, (int, np.integer)):
                trial_params[param_name] = int(round(float(swept_value)))
            else:
                trial_params[param_name] = float(swept_value)
            trials.append(
                {
                    "swept_param": param_name,
                    "swept_value": trial_params[param_name],
                    "params": trial_params,
                }
            )
    return trials


def _setting(config: Mapping[str, Any], key: str, defaults: Mapping[str, Any]) -> Any:
    """Read a run setting, falling back to the corresponding central default."""

    return config[key] if key in config else defaults[key]


def _resolve_output_path(config: Mapping[str, Any], key: str, defaults: Mapping[str, Any]) -> Path:
    """Resolve a configured result path below the configured output directory."""

    configured_path = Path(_setting(config, key, defaults))
    if configured_path.is_absolute():
        return configured_path

    output_dir = Path(config.get("output_dir", RUN_CONFIG["output_dir"]))
    # Central defaults historically include the ``output/`` prefix. Strip that
    # prefix before applying a custom output directory.
    if configured_path.parts and configured_path.parts[0] == "output":
        configured_path = Path(*configured_path.parts[1:])
    return output_dir / configured_path


def _seed_params(params: Mapping[str, Any], seed: int) -> dict[str, Any]:
    """Copy trial parameters and set the seed consumed by the controller."""

    trial_params = dict(params)
    trial_params["seed"] = int(seed)
    return trial_params


def run_gatekeeper_for_seed(
    seed: int,
    params: Mapping[str, Any],
    offline_dataset: Any,
    pcis_data: Mapping[str, Any],
    initial_state: Any | None = None,
    is_first_seed: bool = False,
) -> dict[str, Any]:
    """Run one DRS-GK trial and return serialisable result statistics."""

    if initial_state is None:
        initial_state = state_0

    key = jax.random.PRNGKey(int(seed))
    key, shared_env_key = jax.random.split(key)
    is_open_loop = params.get("open_loop", gk_params["open_loop"])
    if is_open_loop:
        k_trials = int(params.get("K_trials", gk_params["K_trials"]))
        gk_keys = jax.random.split(key, k_trials)
        all_pred_probs = []
        selected_switching_times = []
        for trial_key in gk_keys:
            switch_offset, q_bar_array = get_drs_gk_bounds(
                trial_key,
                initial_state,
                offline_dataset,
                params,
                OBSTACLES,
                GATEKEEPER_VISUAL_RADIUS,
                pcis_data=pcis_data,
            )
            all_pred_probs.append(q_bar_array)
            selected_switching_times.append(int(switch_offset))

        pred_probs = np.array(all_pred_probs)
        k_trials, horizon = pred_probs.shape
        mc_params = dict(params.get("monte_carlo_params", monte_carlo_params))
        mc_params["initial_state"] = initial_state
        mc_params["switching_times"] = list(range(horizon))
        mc_params["seed"] = int(seed)
        mc_params["true_beta"] = float(params.get("true_beta", params.get("beta", TRUE_BETA)))
        plot_override_s = (horizon // 2) if is_first_seed else -1
        mc_results = run_true_probability_monte_carlo(
            params=mc_params,
            dataset=offline_dataset,
            plot_override_s=plot_override_s,
            verbose=False,
        )
        true_probs = np.array([mc_results[s]["prob_failure"] for s in range(horizon)])
        epsilon = float(params["epsilon"])
        valid_selection_count = 0
        correct_selection_count = 0
        correct_gaps = []
        for s_star in selected_switching_times:
            if 0 <= s_star < horizon:
                valid_selection_count += 1
                if true_probs[s_star] <= epsilon:
                    correct_selection_count += 1
                    correct_gaps.append(epsilon - float(true_probs[s_star]))

        empirical_selection_correctness = (
            float(correct_selection_count / valid_selection_count) if valid_selection_count > 0 else float("nan")
        )
        average_correct_gap = float(np.mean(correct_gaps)) if correct_selection_count > 0 else float("nan")
        correctness_matrix = (pred_probs >= true_probs).astype(int)
        marginal_confidence = np.mean(correctness_matrix, axis=0)
        trial_joint_correct = np.all(correctness_matrix == 1, axis=1)
        joint_confidence = float(np.mean(trial_joint_correct))
        delta = float(params["delta"])
        rho = 1.0 - (1.0 - delta) ** (1.0 / float(horizon))
        marginal_threshold = 1.0 - rho
        joint_threshold = 1.0 - delta
        return {
            "K_trials": k_trials,
            "valid_selection_count": valid_selection_count,
            "correct_selection_count": correct_selection_count,
            "empirical_selection_correctness": empirical_selection_correctness,
            "average_correct_gap": average_correct_gap,
            "selected_switching_times_json": json.dumps(selected_switching_times),
            "marginal_confidence_min": float(np.min(marginal_confidence)),
            "marginal_confidence_json": json.dumps(marginal_confidence.tolist()),
            "joint_confidence": joint_confidence,
            "marginal_verified": bool(np.all(marginal_confidence >= marginal_threshold)),
            "joint_verified": bool(joint_confidence >= joint_threshold),
            "predicted_probs_json": json.dumps(pred_probs.tolist()),
            "true_probs_json": json.dumps(true_probs.tolist()),
        }

    key, gk_internal_key = jax.random.split(key)
    _, _, _, _, _, stats = run_gatekeeper_simulation(
        gk_internal_key,
        shared_env_key,
        initial_state,
        offline_dataset,
        params,
        OBSTACLES,
        GATEKEEPER_VISUAL_RADIUS,
        rollout_plot_count=0,
        verbose=False,
        return_stats=True,
        pcis_data=pcis_data,
    )
    return stats


def save_sweep_results_csv(results: list[Mapping[str, Any]], csv_path: str | Path) -> None:
    """Append-independent sweep results in a stable, inspectable CSV schema."""

    if not results:
        return
    keys: set[str] = set()
    for row in results:
        keys.update(row.keys())
    preferred = [
        "algorithm",
        "trial_index",
        "trial_count",
        "seed",
        "swept_param",
        "swept_value",
        "backup_usage_pct",
        "hit_obstacle",
        "infeasible",
        "time_to_goal",
        "runtime_sec",
    ]
    fieldnames = preferred + sorted(key for key in keys if key not in preferred)
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def parse_csv_scalar(value: Any) -> Any:
    """Parse CSV scalars while retaining booleans and missing values."""

    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        lowered = text.lower()
        if lowered == "nan":
            return np.nan
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        try:
            return float(text)
        except ValueError:
            return text
    return value


def signature_scalar(value: Any) -> tuple[str, Any]:
    """Return a hashable, type-aware scalar representation."""

    parsed = parse_csv_scalar(value)
    if parsed is None:
        return ("none", None)
    if isinstance(parsed, (bool, np.bool_)):
        return ("bool", bool(parsed))
    if isinstance(parsed, (int, float, np.integer, np.floating)):
        numeric = float(parsed)
        if np.isnan(numeric):
            return ("nan", None)
        if np.isposinf(numeric):
            return ("inf", 1)
        if np.isneginf(numeric):
            return ("inf", -1)
        return ("num", numeric)
    return ("str", str(parsed))


def params_signature(params: Mapping[str, Any]) -> tuple[tuple[str, tuple[str, Any]], ...]:
    """Build a stable signature used to resume a sweep by parameter set."""

    return tuple(sorted((key, signature_scalar(value)) for key, value in params.items()))


def load_existing_sweep_results(csv_path: str | Path | None, algorithm: str) -> list[dict[str, str]]:
    """Load rows for one algorithm when a prior sweep CSV exists."""

    if not csv_path:
        return []
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8") as fp:
        return [row for row in csv.DictReader(fp) if row.get("algorithm", algorithm) == algorithm]


def build_completed_seed_index(
    results: Iterable[Mapping[str, Any]], param_keys: Iterable[str]
) -> set[tuple[tuple[tuple[str, tuple[str, Any]], ...], int]]:
    """Index completed (trial parameter signature, seed) pairs."""

    completed = set()
    param_keys = list(param_keys)
    for row in results:
        seed_value = parse_csv_scalar(row.get("seed"))
        if seed_value is None:
            continue
        try:
            seed = int(round(float(seed_value)))
        except (TypeError, ValueError):
            continue
        params = {}
        missing = False
        for key in param_keys:
            value = parse_csv_scalar(row.get(f"param_{key}"))
            if value is None:
                missing = True
                break
            params[key] = value
        if not missing:
            completed.add((params_signature(params), seed))
    return completed


def _trial_pcis_data(
    offline_dataset: Any,
    base_pcis_data: Mapping[str, Any],
    params: Mapping[str, Any],
    pcis_key: Any | None,
    pcis_config: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Rebuild PCIS when an epsilon trial requires alpha below its base value."""

    epsilon = float(params["epsilon"])
    base_alpha = float(pcis_config["alpha"])
    if base_alpha < epsilon:
        return base_pcis_data
    if pcis_key is None:
        raise ValueError(
            "An epsilon trial at or below PCIS alpha requires config['pcis_key'] "
            "so the PCIS can be rebuilt with alpha=epsilon/2."
        )
    trial_pcis_params = dict(pcis_config)
    trial_pcis_params["alpha"] = epsilon / 2.0
    trial_pcis_params["true_beta"] = float(params.get("beta", trial_pcis_params.get("true_beta", TRUE_BETA)))
    return _build_pcis_data(
        pcis_key,
        offline_dataset,
        **trial_pcis_params,
    )


def run_one_at_a_time_hyperparameter_sweep(
    config: Mapping[str, Any], offline_dataset: Any, pcis_data: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Run the configured DRS-GK one-at-a-time sweep, resuming completed rows."""

    gk_csv_path = _resolve_output_path(config, "gk_results_csv_path", HYPERPARAM_SWEEP_CONFIG)
    nominal_params = dict(_setting(config, "gk_nominal", HYPERPARAM_SWEEP_CONFIG))
    nominal_params["sensor_noise_scale"] = _setting(config, "sensor_noise_scale", HYPERPARAM_SWEEP_CONFIG)
    sweep_specs = _setting(config, "gk_sweeps", HYPERPARAM_SWEEP_CONFIG)
    trials = build_one_at_a_time_trials(nominal_params, sweep_specs)
    seed_start = int(_setting(config, "seed_start", HYPERPARAM_SWEEP_CONFIG))
    n_seeds = int(_setting(config, "n_seeds", HYPERPARAM_SWEEP_CONFIG))
    existing = load_existing_sweep_results(gk_csv_path, "gatekeeper")
    completed = build_completed_seed_index(existing, nominal_params.keys())

    pcis_config = dict(config.get("pcis_params", DEFAULT_PCIS_PARAMS))
    pcis_key = config.get("pcis_key", pcis_data.get("key"))
    results = list(existing)
    total_trials = len(trials)
    for trial_index, trial in enumerate(trials):
        trial_params = trial["params"]
        signature = params_signature(trial_params)
        pending_seeds = [seed for seed in range(seed_start, seed_start + n_seeds) if (signature, seed) not in completed]
        if not pending_seeds:
            continue
        trial_pcis_data = pcis_data
        if trial["swept_param"] == "epsilon":
            trial_pcis_data = _trial_pcis_data(
                offline_dataset,
                pcis_data,
                trial_params,
                pcis_key,
                pcis_config,
            )
        for seed in pending_seeds:
            seed_offset = seed - seed_start
            seed_params = _seed_params(trial_params, seed)
            stats = run_gatekeeper_for_seed(
                seed,
                seed_params,
                offline_dataset,
                trial_pcis_data,
                is_first_seed=(trial_index == 0 and seed_offset == 0),
            )
            row = {
                "algorithm": "gatekeeper",
                "trial_index": trial_index,
                "trial_count": total_trials,
                "seed": seed,
                "swept_param": trial["swept_param"],
                "swept_value": trial["swept_value"],
                **{f"param_{key}": value for key, value in trial_params.items()},
                **stats,
            }
            results.append(row)
            completed.add((signature, seed))
            save_sweep_results_csv(results, gk_csv_path)
    if results and not existing:
        save_sweep_results_csv(results, gk_csv_path)
    return results


def run_epsilon_beta_sweep(
    config: Mapping[str, Any],
    offline_dataset: Any,
    pcis_key: Any,
    pcis_params: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run one-at-a-time epsilon and beta sweeps for DRS-GK.

    PCIS is rebuilt for every sweep value.  Epsilon values at or below the
    configured PCIS alpha use ``alpha=epsilon/2`` as in the original runner;
    beta values rebuild PCIS with the trial beta as the true-process radius.
    """

    gk_csv_path = _resolve_output_path(config, "gk_results_csv_path", EPS_BETA_SWEEP_CONFIG)
    seed_start = int(_setting(config, "seed_start", EPS_BETA_SWEEP_CONFIG))
    n_seeds = int(_setting(config, "n_seeds", EPS_BETA_SWEEP_CONFIG))
    x_range = tuple(_setting(config, "x_range", EPS_BETA_SWEEP_CONFIG))
    y_range = tuple(_setting(config, "y_range", EPS_BETA_SWEEP_CONFIG))
    avoid_obstacles = bool(_setting(config, "avoid_obstacles", EPS_BETA_SWEEP_CONFIG))
    epsilon_values = _setting(config, "epsilon_values", EPS_BETA_SWEEP_CONFIG)
    beta_values = _setting(config, "beta_values", EPS_BETA_SWEEP_CONFIG)
    nominal_params = dict(HYPERPARAM_SWEEP_CONFIG["gk_nominal"])
    nominal_params["sensor_noise_scale"] = _setting(config, "sensor_noise_scale", EPS_BETA_SWEEP_CONFIG)
    sampled_states = sample_initial_states_in_box(
        n_samples=n_seeds,
        seed=seed_start,
        x_range=x_range,
        y_range=y_range,
        base_state=state_0,
        obstacles=OBSTACLES,
        avoid_obstacles=avoid_obstacles,
    )
    nominal_beta = float(nominal_params["beta"])
    nominal_epsilon = float(nominal_params["epsilon"])
    default_alpha = float(pcis_params["alpha"])
    results: list[dict[str, Any]] = []

    def record_rows(
        trial_index: int,
        trial_count: int,
        param_name: str,
        swept_value: float,
        run_params: Mapping[str, Any],
        trial_pcis_data: Mapping[str, Any],
    ) -> None:
        for seed_offset, (seed, initial_state) in enumerate(
            zip(range(seed_start, seed_start + n_seeds), sampled_states)
        ):
            seed_params = _seed_params(run_params, seed)
            start_time = time.time()
            metrics = run_gatekeeper_for_seed(
                seed,
                seed_params,
                offline_dataset,
                trial_pcis_data,
                initial_state=initial_state,
                is_first_seed=(seed_offset == 0),
            )
            results.append(
                {
                    "algorithm": "gatekeeper",
                    "trial_index": trial_index,
                    "trial_count": trial_count,
                    "seed": int(seed),
                    "swept_param": param_name,
                    "swept_value": swept_value,
                    "runtime_sec": float(time.time() - start_time),
                    **{f"param_{key}": value for key, value in run_params.items()},
                    **metrics,
                }
            )

    epsilon_trial_count = len(epsilon_values)
    for trial_index, epsilon_value in enumerate(epsilon_values):
        epsilon = float(epsilon_value)
        current_alpha = epsilon / 2.0 if default_alpha >= epsilon else default_alpha
        current_pcis_params = dict(pcis_params)
        current_pcis_params["alpha"] = current_alpha
        current_pcis_params["true_beta"] = nominal_beta
        epsilon_pcis_data = _build_pcis_data(
            pcis_key,
            offline_dataset,
            **current_pcis_params,
        )
        run_params = dict(nominal_params)
        run_params["epsilon"] = epsilon
        run_params["beta"] = nominal_beta
        run_params["true_beta"] = nominal_beta
        record_rows(
            trial_index,
            epsilon_trial_count,
            "epsilon",
            epsilon,
            run_params,
            epsilon_pcis_data,
        )

    beta_trial_offset = len(epsilon_values)
    beta_trial_count = beta_trial_offset + len(beta_values)
    for beta_index, beta_value in enumerate(beta_values):
        beta = float(beta_value)
        current_alpha = nominal_epsilon / 2.0 if default_alpha >= nominal_epsilon else default_alpha
        current_pcis_params = dict(pcis_params)
        current_pcis_params["alpha"] = current_alpha
        current_pcis_params["true_beta"] = beta
        beta_pcis_data = _build_pcis_data(
            pcis_key,
            offline_dataset,
            **current_pcis_params,
        )
        run_params = dict(nominal_params)
        run_params["epsilon"] = nominal_epsilon
        run_params["beta"] = beta
        run_params["true_beta"] = beta
        record_rows(
            beta_trial_offset + beta_index,
            beta_trial_count,
            "beta",
            beta,
            run_params,
            beta_pcis_data,
        )

    if results:
        save_sweep_results_csv(results, gk_csv_path)
    return results, []


def summarize_sweep_results(
    results: Iterable[Mapping[str, Any]],
    algorithm_name: str,
    group_param_name: str,
) -> list[dict[str, Any]]:
    """Summarize rows grouped by one swept parameter (epsilon or beta)."""

    groups: dict[Any, list[Mapping[str, Any]]] = {}
    for row in results:
        if str(row.get("algorithm", algorithm_name)) != algorithm_name:
            continue
        if str(row.get("swept_param")) != group_param_name:
            continue
        value = parse_csv_scalar(row.get("swept_value"))
        groups.setdefault(value, []).append(row)

    summaries = []
    for value, rows in sorted(groups.items(), key=lambda item: float(item[0])):
        n_rows = len(rows)
        safeties = []
        hit_rates = []
        infeasible_rates = []
        times = []
        backups = []
        runtimes = []
        for row in rows:
            hit = parse_csv_scalar(row.get("hit_obstacle", row.get("gk_hit_obstacle", row.get("cbf_hit_obstacle"))))
            infeasible = parse_csv_scalar(row.get("infeasible"))
            time_to_goal = parse_csv_scalar(
                row.get("time_to_goal", row.get("gk_time_to_goal", row.get("cbf_time_to_goal")))
            )
            backup_usage = parse_csv_scalar(row.get("backup_usage_pct", row.get("gk_backup_usage_pct")))
            runtime = parse_csv_scalar(row.get("runtime_sec"))
            hit_bool = bool(hit) if hit is not None else False
            infeasible_bool = bool(infeasible) if infeasible is not None else False
            safe = (not hit_bool) and (not infeasible_bool if algorithm_name == "cbf" else True)
            safeties.append(1.0 if safe else 0.0)
            hit_rates.append(1.0 if hit_bool else 0.0)
            infeasible_rates.append(1.0 if infeasible_bool else 0.0)
            times.append(
                float(time_to_goal) if time_to_goal is not None and np.isfinite(float(time_to_goal)) else np.nan
            )
            backups.append(float(backup_usage) if backup_usage is not None else np.nan)
            runtimes.append(float(runtime) if runtime is not None else np.nan)

        summaries.append(
            {
                group_param_name: float(value),
                "n_runs": n_rows,
                "safety_rate": float(np.nanmean(safeties)) if n_rows > 0 else 0.0,
                "hit_rate": float(np.nanmean(hit_rates)) if n_rows > 0 else 0.0,
                "infeasible_rate": (float(np.nanmean(infeasible_rates)) if n_rows > 0 else 0.0),
                "mean_time_to_goal": (
                    float(np.nanmean(times)) if any(not np.isnan(value) for value in times) else float(np.inf)
                ),
                "mean_backup_usage_pct": (
                    float(np.nanmean(backups)) if any(not np.isnan(value) for value in backups) else 0.0
                ),
                "mean_runtime_sec": (
                    float(np.nanmean(runtimes)) if any(not np.isnan(value) for value in runtimes) else 0.0
                ),
            }
        )
    return summaries


def save_summary_csv(summary_rows: Iterable[Mapping[str, Any]], csv_path: str | Path) -> None:
    """Write summary rows to a CSV for downstream analysis."""

    rows = list(summary_rows)
    if not rows:
        return
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
