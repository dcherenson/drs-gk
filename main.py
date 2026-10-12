"""Run controller comparisons and optional experiments from the repository root."""

import argparse
import copy
import csv
from pathlib import Path

import jax
import numpy as np

import config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=["compare", "initial-states", "sweep", "eps-beta", "monte-carlo"],
        default=config.RUN_CONFIG["experiment"],
    )
    parser.add_argument("--controllers", nargs="+", choices=list(config.CONTROLLER_STYLES))
    parser.add_argument(
        "--quick", action="store_true", help="Run a DRS-GK goal demo, or use reduced experiment settings."
    )
    parser.add_argument("--steps", type=int, help="Override the comparison simulation length.")
    parser.add_argument("--seed", type=int, help="Override the experiment seed.")
    parser.add_argument("--output-dir", type=Path, help="Override the output directory.")
    parser.add_argument("--no-animation", action="store_true", help="Skip GIF generation.")
    parser.add_argument(
        "--verbose", action="store_true", default=config.RUN_CONFIG["verbose"], help="Print controller progress."
    )
    args = parser.parse_args()
    if args.quick:
        quick_controllers = config.QUICK_CONFIG["controllers"]
        if args.controllers is not None and any(name not in quick_controllers for name in args.controllers):
            parser.error("--quick runs DRS-GK only; omit --quick to compare controllers")
        args.controllers = quick_controllers.copy()
    elif args.controllers is None:
        args.controllers = config.RUN_CONFIG["controllers"].copy()
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be nonnegative")
    return args


def configure_run(args):
    """Apply overrides before importing modules that compile JAX functions."""
    if args.quick:
        quick = config.QUICK_CONFIG
        config.N_STEPS = quick["steps"]
        config.gk_params.update(quick["gk"])
        if args.experiment in ("compare", "initial-states"):
            config.gk_params.update(quick["demo_gk"])
        config.pcis_params.update(quick["pcis"])
        config.monte_carlo_params.update(quick["monte_carlo"])
        config.INITIAL_STATE_COMPARISON_CONFIG["n_samples"] = quick["initial_states"]
        config.HYPERPARAM_SWEEP_CONFIG["n_seeds"] = quick["sweep_n_seeds"]
        config.EPS_BETA_SWEEP_CONFIG["n_seeds"] = quick["sweep_n_seeds"]
        config.HYPERPARAM_SWEEP_CONFIG["gk_sweeps"] = copy.deepcopy(quick["gk_sweeps"])
        config.HYPERPARAM_SWEEP_CONFIG["gk_nominal"].update(quick["gk"])
        config.EPS_BETA_SWEEP_CONFIG["epsilon_values"] = quick["epsilon_values"]
        config.EPS_BETA_SWEEP_CONFIG["beta_values"] = quick["beta_values"]
    if args.steps is not None:
        config.N_STEPS = args.steps
    if args.seed is not None:
        config.RUN_CONFIG["seed"] = args.seed
        config.INITIAL_STATE_COMPARISON_CONFIG["seed"] = args.seed
        config.HYPERPARAM_SWEEP_CONFIG["seed_start"] = args.seed
        config.EPS_BETA_SWEEP_CONFIG["seed_start"] = args.seed
        config.monte_carlo_params["seed"] = args.seed
    output = (args.output_dir or Path(config.RUN_CONFIG["output_dir"])).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config.RUN_CONFIG["output_dir"] = str(output)
    config.monte_carlo_params["output_dir"] = str(output)
    for settings in (config.HYPERPARAM_SWEEP_CONFIG, config.EPS_BETA_SWEEP_CONFIG):
        for key in ("gk_results_csv_path", "cbf_results_csv_path"):
            if key in settings:
                settings[key] = str(output / Path(settings[key]).name)
    return output


def save_metrics(rows, path):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_comparison(initial_state, dataset, pcis_data, seed, controllers, verbose=False, sensor_noise_scale=None):
    key = jax.random.PRNGKey(seed)
    env_key, gk_key, smps_key, mpc_key = jax.random.split(key, 4)
    results = {}
    common = (initial_state, dataset, config.OBSTACLES)
    gk_options = config.gk_params.copy()
    cbf_options = config.cbf_params.copy()
    sensor_scale = config.SENSOR_NOISE_SCALE if sensor_noise_scale is None else sensor_noise_scale
    cbf_options.update(sensor_noise_scale=sensor_scale, sensor_noise_shape=config.SENSOR_NOISE_SHAPE)
    if sensor_noise_scale is not None:
        gk_options["sensor_noise_scale"] = sensor_noise_scale
    for name in controllers:
        print(f"Running {name}...")
        if name == "drs-gk":
            from controller.drs_gatekeeper import run_gatekeeper_simulation

            states, controls, _, _, _, stats = run_gatekeeper_simulation(
                gk_key,
                env_key,
                initial_state,
                dataset,
                gk_options,
                config.OBSTACLES,
                config.GATEKEEPER_VISUAL_RADIUS,
                rollout_plot_count=0,
                verbose=verbose,
                return_stats=True,
                pcis_data=pcis_data,
            )
        elif name == "gatekeeper":
            from controller.gatekeeper import run_vanilla_gatekeeper_simulation

            states, controls, _, _, stats = run_vanilla_gatekeeper_simulation(
                env_key,
                initial_state,
                dataset,
                gk_options,
                config.OBSTACLES,
                config.GATEKEEPER_VISUAL_RADIUS,
                verbose=verbose,
                return_stats=True,
            )
        elif name == "smps":
            from controller.smps import run_smps_simulation

            states, controls, _, _, stats = run_smps_simulation(
                smps_key,
                env_key,
                initial_state,
                dataset,
                gk_options,
                config.OBSTACLES,
                config.GATEKEEPER_VISUAL_RADIUS,
                pcis_data=pcis_data,
                rollout_plot_count=0,
                verbose=verbose,
                return_stats=True,
            )
        elif name == "dr-cbf":
            from controller.dr_cbf import run_cbf_simulation

            states, controls, stats = run_cbf_simulation(
                env_key,
                *common,
                params=cbf_options,
                verbose=verbose,
                return_stats=True,
            )
        else:
            from controller.wt_mpc import run_wt_mpc_simulation

            states, controls, stats = run_wt_mpc_simulation(
                mpc_key,
                env_key,
                *common,
                config.wt_mpc_params,
                n_steps=config.N_STEPS,
                verbose=verbose,
                return_stats=True,
                sensor_noise_scale=sensor_scale,
                sensor_noise_shape=config.SENSOR_NOISE_SHAPE,
            )
        goal_distance = float(np.linalg.norm(np.asarray(states[-1, :2]) - np.asarray(config.target_pos)))
        stats = dict(
            stats,
            final_goal_distance=goal_distance,
            goal_reached=bool(np.isfinite(stats.get("time_to_goal", np.inf))),
            steps=int(len(controls)),
        )
        results[name] = {"states": states, "controls": controls, "stats": stats}
        status = (
            "collision"
            if stats.get("hit_obstacle")
            else "infeasible"
            if stats.get("infeasible")
            else "goal reached"
            if stats["goal_reached"]
            else "finished"
        )
        print(f"  {status}; {stats['steps']} steps; goal distance {goal_distance:.2f} m")
    return results


def run_initial_states(dataset, pcis_data, args, output):
    from helper import collides_with_any_obstacle

    options = config.INITIAL_STATE_COMPARISON_CONFIG
    csv_path = output / options["csv_filename"]
    rng = np.random.default_rng(options["seed"])
    rows = []
    for index in range(options["n_samples"]):
        while True:
            initial = config.state_0.at[0].set(rng.uniform(*options["x_range"]))
            initial = initial.at[1].set(rng.uniform(*options["y_range"]))
            if not options.get("avoid_obstacles", False) or not collides_with_any_obstacle(initial, config.OBSTACLES):
                break
        results = run_comparison(
            initial,
            dataset,
            pcis_data,
            options["seed"],
            args.controllers,
            args.verbose,
            sensor_noise_scale=options["sensor_noise_scale"],
        )
        for name, result in results.items():
            rows.append(
                dict(
                    sample_idx=index,
                    controller=name,
                    seed=options["seed"],
                    initial_x=float(initial[0]),
                    initial_y=float(initial[1]),
                    **result["stats"],
                )
            )
        save_metrics(rows, csv_path)
    print(f"Saved {len(rows)} controller runs to {csv_path}")


def main():
    args = parse_args()
    output = configure_run(args)
    from helper import generate_offline_dataset

    print(f"JAX backend: {jax.default_backend()} ({jax.devices()[0]})")
    if args.quick:
        if args.experiment in ("compare", "initial-states"):
            print("Using DRS-GK goal-demo settings from config.py.")
        else:
            print("Using reduced experiment settings from config.py.")
    if args.experiment == "monte-carlo":
        from experiments.monte_carlo_analysis import run_true_probability_monte_carlo

        run_true_probability_monte_carlo(params=config.monte_carlo_params.copy())
        return
    key = jax.random.PRNGKey(config.RUN_CONFIG["seed"])
    dataset_key, pcis_key = jax.random.split(key)
    dataset = generate_offline_dataset(dataset_key)
    needs_pcis = args.experiment in ("sweep", "eps-beta") or (
        args.experiment in ("compare", "initial-states")
        and any(name in args.controllers for name in ("drs-gk", "smps"))
    )
    pcis_data = None
    if needs_pcis:
        from pcis import build_pcis_data

        pcis_data = build_pcis_data(pcis_key, dataset, **config.pcis_params)
    if args.experiment == "compare":
        from visualization import animate_comparison, plot_comparison

        results = run_comparison(
            config.state_0, dataset, pcis_data, config.RUN_CONFIG["seed"], args.controllers, args.verbose
        )
        save_metrics(
            [
                dict(controller=name, seed=config.RUN_CONFIG["seed"], **result["stats"])
                for name, result in results.items()
            ],
            output / config.RUN_CONFIG["metrics_filename"],
        )
        plot_comparison(results, config.OBSTACLES, config.target_pos, output / config.RUN_CONFIG["plot_filename"])
        if config.RUN_CONFIG["animation"] and not args.no_animation:
            animate_comparison(
                results, config.OBSTACLES, config.target_pos, output / config.RUN_CONFIG["animation_filename"]
            )
        print(f"Saved comparison outputs to {output}")
    elif args.experiment == "initial-states":
        run_initial_states(dataset, pcis_data, args, output)
    else:
        from experiments.sweeps import (
            run_one_at_a_time_hyperparameter_sweep,
            run_epsilon_beta_sweep,
            summarize_sweep_results,
            save_summary_csv,
        )

        if args.experiment == "sweep":
            sweep_config = dict(config.HYPERPARAM_SWEEP_CONFIG, pcis_key=pcis_key)
            run_one_at_a_time_hyperparameter_sweep(sweep_config, dataset, pcis_data)
        else:
            rows, _ = run_epsilon_beta_sweep(config.EPS_BETA_SWEEP_CONFIG, dataset, pcis_key, config.pcis_params)
            for parameter in ("epsilon", "beta"):
                save_summary_csv(
                    summarize_sweep_results(rows, "gatekeeper", parameter), output / f"{parameter}_summary_gk.csv"
                )
        print(f"Saved experiment outputs to {output}")


if __name__ == "__main__":
    main()
