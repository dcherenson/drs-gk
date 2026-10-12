import argparse
import csv
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from config import (
    EXCLUDED_PLOT_PARAMS,
    HYPERPARAM_PLOT_CONFIG,
    HYPERPARAM_SWEEP_CONFIG,
    LOG_SCALE_PARAMS,
    METRIC_SPECS,
    PLOT_DPI,
    PLOT_PARAM_LABELS,
    RUN_CONFIG,
    SHOW_ERROR_BARS,
)

plt.switch_backend("Agg")
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "DejaVu Sans"


def parse_float(value):
    if value is None:
        return math.nan
    s = str(value).strip()
    if not s:
        return math.nan
    try:
        return float(s)
    except ValueError:
        return math.nan


def parse_bool(value):
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in {"true", "1", "yes"}:
        return True
    if s in {"false", "0", "no"}:
        return False
    return None


def format_pct(x):
    if x is None or math.isnan(x):
        return "N/A"
    return f"{x:6.2f}%"


def format_seconds(x):
    if x is None or math.isnan(x):
        return "N/A"
    return f"{x:8.3f}"


def print_table(title, rows):
    headers = [
        "Sweep",
        "Seeds",
        "Avg Goal Reach Rate",
        "Avg Safety Rate",
        "Avg Time To Goal (s)",
        "Avg Backup Pct",
    ]

    printable_rows = [
        [
            format_sweep_label(row["swept_param"], row["swept_value"]),
            str(row["seed_count"]),
            format_pct(row["goal_reach_rate_pct"]),
            format_pct(row["safety_rate_pct"]),
            format_seconds(row["avg_time_to_goal_sec"]),
            format_pct(row["avg_backup_pct"]),
        ]
        for row in rows
    ]

    widths = [len(h) for h in headers]
    for row in printable_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    sep = "-+-".join("-" * w for w in widths)

    print(f"\n{title}")
    print(" | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print(sep)
    for row in printable_rows:
        print(" | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))


def parse_trial_index(value):
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def format_sweep_label(swept_param, swept_value):
    param = (swept_param or "").strip()
    value = parse_float(swept_value)
    if param == "nominal" or math.isnan(value):
        return "nominal"
    if float(value).is_integer():
        return f"{param}={int(value)}"
    return f"{param}={value:.6g}"


def format_plot_param_label(param_name):
    return PLOT_PARAM_LABELS.get(param_name, param_name)


def compute_percentile(values, percentile):
    if not values:
        return math.nan
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = percentile * (len(sorted_values) - 1)
    lower_idx = int(math.floor(position))
    upper_idx = int(math.ceil(position))
    lower_value = sorted_values[lower_idx]
    upper_value = sorted_values[upper_idx]
    weight = position - lower_idx
    return lower_value + (weight * (upper_value - lower_value))


def time_to_goal_score(time_value, fastest_time):
    if not math.isfinite(time_value):
        return 0.0
    if fastest_time <= 0.0 or time_value <= 0.0:
        return 100.0
    return max(0.0, min(100.0, 100.0 * fastest_time / time_value))


def build_summary(rows):
    # Group by trial so each average is across that trial's seed runs.
    grouped = {}
    trial_order = []
    for row in rows:
        trial_index = parse_trial_index(row.get("trial_index"))
        swept_param = (row.get("swept_param") or "").strip()
        swept_value = row.get("swept_value")
        key = (trial_index, swept_param, str(swept_value))
        if key not in grouped:
            grouped[key] = []
            trial_order.append(key)
        grouped[key].append(row)

    trial_order.sort(
        key=lambda k: (
            1 if k[0] is None else 0,
            float("inf") if k[0] is None else k[0],
            k[1],
            k[2],
        )
    )

    summary = []
    for trial_index, swept_param, swept_value in trial_order:
        group_rows = grouped[(trial_index, swept_param, swept_value)]
        n = len(group_rows)
        if n == 0:
            continue

        sample_row = group_rows[0]
        param_values = {}
        for key, value in sample_row.items():
            if key.startswith("param_"):
                param_values[key[6:]] = parse_float(value)

        goal_hits = 0
        safe_runs = 0
        goal_times = []
        time_values = []
        backup_values = []

        for row in group_rows:
            t_goal = parse_float(row.get("time_to_goal"))
            hit_obstacle = parse_bool(row.get("hit_obstacle"))
            infeasible = parse_bool(row.get("infeasible"))
            backup_pct = parse_float(row.get("backup_usage_pct"))
            time_values.append(t_goal)

            reached_goal = math.isfinite(t_goal)
            if reached_goal:
                goal_hits += 1
                goal_times.append(t_goal)

            is_safe = (hit_obstacle is not True) and (infeasible is not True)
            if is_safe:
                safe_runs += 1

            if math.isfinite(backup_pct):
                backup_values.append(backup_pct)

        summary.append(
            {
                "trial_index": trial_index,
                "swept_param": swept_param,
                "swept_value": parse_float(swept_value),
                "seed_count": n,
                "goal_reach_rate_pct": 100.0 * goal_hits / n,
                "safety_rate_pct": 100.0 * safe_runs / n,
                "avg_time_to_goal_sec": sum(goal_times) / len(goal_times) if goal_times else math.nan,
                "avg_backup_pct": sum(backup_values) / len(backup_values) if backup_values else math.nan,
                "time_to_goal_values": time_values,
                "backup_pct_values": backup_values,
                "param_values": param_values,
            }
        )
    return summary


def ordered_swept_params(summary_rows):
    params = []
    for row in summary_rows:
        swept_param = row["swept_param"]
        if swept_param == "nominal" or swept_param in EXCLUDED_PLOT_PARAMS or swept_param in params:
            continue
        params.append(swept_param)

    epsilon_name = "epsilon" if "epsilon" in params else ("eps" if "eps" in params else None)
    if "beta" in params and epsilon_name in params:
        beta_idx = params.index("beta")
        epsilon_idx = params.index(epsilon_name)
        params[beta_idx], params[epsilon_idx] = params[epsilon_idx], params[beta_idx]

    return params


def add_normalized_time_metric(summary_rows):
    finite_times = [
        time_value
        for row in summary_rows
        for time_value in row.get("time_to_goal_values", [])
        if math.isfinite(time_value)
    ]
    fastest_time = min(finite_times) if finite_times else math.nan

    normalized_rows = []
    for row in summary_rows:
        normalized_row = dict(row)
        time_scores = []
        if math.isfinite(fastest_time):
            time_scores = [
                time_to_goal_score(time_value, fastest_time) for time_value in row.get("time_to_goal_values", [])
            ]

        if time_scores:
            normalized_score = sum(time_scores) / len(time_scores)
            time_lower = compute_percentile(time_scores, 0.025)
            time_upper = compute_percentile(time_scores, 0.975)
            normalized_row["normalized_avg_time_to_goal_pct"] = normalized_score
            normalized_row["normalized_avg_time_to_goal_lower_err"] = max(0.0, normalized_score - time_lower)
            normalized_row["normalized_avg_time_to_goal_upper_err"] = max(0.0, time_upper - normalized_score)
        else:
            normalized_row["normalized_avg_time_to_goal_pct"] = 0.0
            normalized_row["normalized_avg_time_to_goal_lower_err"] = 0.0
            normalized_row["normalized_avg_time_to_goal_upper_err"] = 0.0

        backup_values = row.get("backup_pct_values", [])
        if backup_values and math.isfinite(row["avg_backup_pct"]):
            backup_lower = compute_percentile(backup_values, 0.025)
            backup_upper = compute_percentile(backup_values, 0.975)
            normalized_row["avg_backup_pct_lower_err"] = max(0.0, row["avg_backup_pct"] - backup_lower)
            normalized_row["avg_backup_pct_upper_err"] = max(0.0, backup_upper - row["avg_backup_pct"])
        else:
            normalized_row["avg_backup_pct_lower_err"] = 0.0
            normalized_row["avg_backup_pct_upper_err"] = 0.0
        normalized_rows.append(normalized_row)
    return normalized_rows


def build_param_plot_series(summary_rows, swept_param):
    points = []
    error_keys = [
        "normalized_avg_time_to_goal_lower_err",
        "normalized_avg_time_to_goal_upper_err",
        "avg_backup_pct_lower_err",
        "avg_backup_pct_upper_err",
    ]

    nominal_row = next((row for row in summary_rows if row["swept_param"] == "nominal"), None)
    if nominal_row is not None:
        nominal_value = nominal_row["param_values"].get(swept_param, math.nan)
        if math.isfinite(nominal_value):
            points.append(
                {
                    "x": nominal_value,
                    **{metric_key: nominal_row[metric_key] for metric_key, _ in METRIC_SPECS},
                    **{error_key: nominal_row.get(error_key, 0.0) for error_key in error_keys},
                }
            )

    for row in summary_rows:
        if row["swept_param"] != swept_param:
            continue
        swept_value = row["swept_value"]
        if not math.isfinite(swept_value):
            continue
        points.append(
            {
                "x": swept_value,
                **{metric_key: row[metric_key] for metric_key, _ in METRIC_SPECS},
                **{error_key: row.get(error_key, 0.0) for error_key in error_keys},
            }
        )

    points.sort(key=lambda point: point["x"])
    return points


def build_plot_output_path(csv_file, algorithm_label, output_dir=None):
    csv_path = Path(csv_file)
    safe_label = (algorithm_label or "all").strip().replace(" ", "_")
    if output_dir is None:
        output_dir = csv_path.parent if csv_path.parent != Path(".") else RUN_CONFIG["output_dir"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{csv_path.stem}_{safe_label}_metrics.svg"


def configure_x_axis(ax, swept_param, x_values):
    if not x_values:
        return

    unique_x = sorted(set(x_values))
    if swept_param in LOG_SCALE_PARAMS:
        positive_x = [x for x in unique_x if x > 0]
        if positive_x:
            if any(x <= 0 for x in unique_x):
                ax.set_xscale("symlog", linthresh=min(positive_x) / 10.0)
            else:
                ax.set_xscale("log")

    ax.set_xticks(unique_x)
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    formatter.set_useOffset(False)
    ax.xaxis.set_major_formatter(formatter)


def plot_hyperparameter_sweep(summary_rows, csv_file, algorithm_label):
    summary_rows = add_normalized_time_metric(summary_rows)
    swept_params = ordered_swept_params(summary_rows)
    if not swept_params:
        print(f"No swept hyperparameters found to plot for {algorithm_label}.")
        return

    n_plots = len(swept_params)
    n_cols = HYPERPARAM_PLOT_CONFIG["n_cols"]
    n_rows = HYPERPARAM_PLOT_CONFIG["n_rows"]
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(
            HYPERPARAM_PLOT_CONFIG["subplot_width"] * n_cols,
            HYPERPARAM_PLOT_CONFIG["subplot_height"] * n_rows,
        ),
        squeeze=False,
    )

    colors = HYPERPARAM_PLOT_CONFIG["colors"]
    markers = HYPERPARAM_PLOT_CONFIG["markers"]
    label_fontsize = HYPERPARAM_PLOT_CONFIG["label_fontsize"]
    tick_fontsize = HYPERPARAM_PLOT_CONFIG["tick_fontsize"]
    legend_fontsize = HYPERPARAM_PLOT_CONFIG["legend_fontsize"]

    for ax, swept_param in zip(axes.flat, swept_params):
        series = build_param_plot_series(summary_rows, swept_param)
        x_values = [point["x"] for point in series]

        for idx, (metric_key, metric_label) in enumerate(METRIC_SPECS):
            y_values = [point[metric_key] for point in series]
            if SHOW_ERROR_BARS and metric_key == "avg_backup_pct":
                lower_err = [point.get("avg_backup_pct_lower_err", 0.0) for point in series]
                upper_err = [point.get("avg_backup_pct_upper_err", 0.0) for point in series]
                ax.errorbar(
                    x_values,
                    y_values,
                    yerr=[lower_err, upper_err],
                    marker=markers[idx],
                    color=colors[idx],
                    linewidth=HYPERPARAM_PLOT_CONFIG["line_width"],
                    elinewidth=HYPERPARAM_PLOT_CONFIG["errorbar_line_width"],
                    capsize=HYPERPARAM_PLOT_CONFIG["errorbar_capsize"],
                    label=metric_label,
                )
            elif SHOW_ERROR_BARS and metric_key == "normalized_avg_time_to_goal_pct":
                lower_err = [point.get("normalized_avg_time_to_goal_lower_err", 0.0) for point in series]
                upper_err = [point.get("normalized_avg_time_to_goal_upper_err", 0.0) for point in series]
                ax.errorbar(
                    x_values,
                    y_values,
                    yerr=[lower_err, upper_err],
                    marker=markers[idx],
                    color=colors[idx],
                    linewidth=HYPERPARAM_PLOT_CONFIG["line_width"],
                    elinewidth=HYPERPARAM_PLOT_CONFIG["errorbar_line_width"],
                    capsize=HYPERPARAM_PLOT_CONFIG["errorbar_capsize"],
                    label=metric_label,
                )
            else:
                ax.plot(
                    x_values,
                    y_values,
                    marker=markers[idx],
                    color=colors[idx],
                    linewidth=HYPERPARAM_PLOT_CONFIG["line_width"],
                    label=metric_label,
                )

        configure_x_axis(ax, swept_param, x_values)
        ax.set_xlabel(format_plot_param_label(swept_param), fontsize=label_fontsize)
        ax.set_ylabel(HYPERPARAM_PLOT_CONFIG["y_label"], fontsize=label_fontsize)
        ax.set_ylim(*HYPERPARAM_PLOT_CONFIG["y_limits"])
        ax.set_yticks(HYPERPARAM_PLOT_CONFIG["y_ticks"])
        ax.tick_params(axis="both", labelsize=tick_fontsize)
        ax.tick_params(axis="x", labelrotation=HYPERPARAM_PLOT_CONFIG["x_tick_rotation"])
        ax.grid(True, alpha=HYPERPARAM_PLOT_CONFIG["grid_alpha"])

    for ax in axes.flat[n_plots:]:
        ax.set_visible(False)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc=HYPERPARAM_PLOT_CONFIG["legend_loc"],
        ncol=HYPERPARAM_PLOT_CONFIG["legend_ncol"],
        frameon=HYPERPARAM_PLOT_CONFIG["legend_frameon"],
        bbox_to_anchor=HYPERPARAM_PLOT_CONFIG["legend_bbox_to_anchor"],
        fontsize=legend_fontsize,
    )
    fig.tight_layout(rect=HYPERPARAM_PLOT_CONFIG["tight_layout_rect"])

    output_path = build_plot_output_path(csv_file, algorithm_label)
    fig.savefig(
        output_path,
        dpi=PLOT_DPI,
        bbox_inches=HYPERPARAM_PLOT_CONFIG["bbox_inches"],
    )
    plt.close(fig)
    print(f"Saved hyperparameter sweep plot to {output_path}")


def analyze_hyperparameter_sweep(csv_file):
    with open(csv_file, newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))

    if not rows:
        print("No rows found in CSV.")
        return

    algorithms = sorted({(row.get("algorithm") or "").strip() for row in rows if (row.get("algorithm") or "").strip()})

    if len(algorithms) <= 1:
        summary = build_summary(rows)
        algo_label = algorithms[0] if algorithms else "all"
        print_table(f"Hyperparameter Sweep Summary ({algo_label})", summary)
        plot_hyperparameter_sweep(summary, csv_file, algo_label)
        return

    for algo in algorithms:
        algo_rows = [row for row in rows if (row.get("algorithm") or "").strip() == algo]
        summary = build_summary(algo_rows)
        print_table(f"Hyperparameter Sweep Summary ({algo})", summary)
        plot_hyperparameter_sweep(summary, csv_file, algo)


def main():
    parser = argparse.ArgumentParser(description="Print and plot a hyperparameter sweep summary from CSV.")
    parser.add_argument(
        "csv_file",
        nargs="?",
        default=str(HYPERPARAM_SWEEP_CONFIG["gk_results_csv_path"]),
        help="Path to sweep results CSV (default: output/hyperparam_sweep_results_gk.csv)",
    )
    args = parser.parse_args()
    analyze_hyperparameter_sweep(args.csv_file)


if __name__ == "__main__":
    main()
