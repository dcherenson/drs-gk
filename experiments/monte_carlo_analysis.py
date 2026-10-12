import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from config import (
    ANIMATION_PADDING,
    MONTE_CARLO_PLOT_CONFIG,
    OBSTACLES,
    PLOT_DPI,
    PLOT_SEED,
    monte_carlo_params,
    target_pos,
    target_v,
)
from helper import (
    nominal_policy,
    format_obstacles,
    generate_offline_dataset,
    sample_coupled_true_disturbance_sequences,
    rollout_bundle,
    add_obstacles_to_axis,
)


def batch_check_collisions(states, obstacles_np):
    """
    Evaluates whether any state in each rollout trajectory collides with any obstacle.

    Args:
        states: Array of shape (n_runs, horizon + 1, state_dim)
        obstacles_np: Array of shape (n_obstacles, 3) [center_x, center_y, radius]

    Returns:
        Boolean array of shape (n_runs,) indicating collision for each rollout.
    """
    positions = np.array(states[..., :2])  # (n_runs, T+1, 2)
    centers = obstacles_np[:, :2]  # (n_obs, 2)
    radii = obstacles_np[:, 2]  # (n_obs,)

    # Compute distances to all obstacle centers across all runs and timesteps
    diff = positions[:, :, None, :] - centers[None, None, :, :]
    distances = np.linalg.norm(diff, axis=-1)  # (n_runs, T+1, n_obs)
    clearances = distances - radii[None, None, :]  # (n_runs, T+1, n_obs)

    # Collision occurs if clearance <= 0 for any obstacle at any time
    collides_step = np.any(clearances <= 0.0, axis=-1)  # (n_runs, T+1)
    collides_run = np.any(collides_step, axis=-1)  # (n_runs,)
    return collides_run


def plot_mc_rollouts(s, rollout_states, collisions, obstacles_np, params):
    n_plot_lines = params.get("n_plot_lines", monte_carlo_params["n_plot_lines"])
    target_pos_np = np.array(target_pos)

    # Sample lines if there are too many
    total_runs = rollout_states.shape[0]
    if total_runs > n_plot_lines:
        np.random.seed(int(params.get("plot_seed", PLOT_SEED)))
        indices = np.random.choice(total_runs, size=n_plot_lines, replace=False)
        plot_states = rollout_states[indices]
        plot_collisions = collisions[indices]
    else:
        plot_states = rollout_states
        plot_collisions = collisions

    fig, ax = plt.subplots(figsize=MONTE_CARLO_PLOT_CONFIG["figsize"])
    add_obstacles_to_axis(ax, obstacles_np)
    ax.plot(
        target_pos_np[0],
        target_pos_np[1],
        marker="*",
        color="gold",
        markersize=15,
        markeredgecolor="black",
        label="Goal",
        zorder=10,
    )

    start_pos = plot_states[0, 0, :2]
    ax.plot(
        start_pos[0],
        start_pos[1],
        marker="s",
        color="black",
        markersize=8,
        label="Start",
        zorder=10,
    )

    for i, traj in enumerate(plot_states):
        has_failed = plot_collisions[i]

        if has_failed:
            dists = (
                np.linalg.norm(traj[:, np.newaxis, :2] - obstacles_np[np.newaxis, :, :2], axis=2)
                - obstacles_np[np.newaxis, :, 2]
            )
            is_collision = np.any(dists <= 0, axis=1)
            fail_idx = np.argmax(is_collision)
        else:
            fail_idx = traj.shape[0]

        nom_end = min(s + 1, fail_idx + 1 if has_failed else s + 1)
        nominal_traj = traj[:nom_end, :2]
        if nominal_traj.shape[0] > 0:
            ax.plot(
                nominal_traj[:, 0],
                nominal_traj[:, 1],
                color="blue",
                alpha=MONTE_CARLO_PLOT_CONFIG["trajectory_alpha"],
                linewidth=MONTE_CARLO_PLOT_CONFIG["trajectory_linewidth"],
            )

        if s < traj.shape[0] and s <= fail_idx:
            back_end = fail_idx + 1 if has_failed else traj.shape[0]
            backup_traj = traj[s:back_end, :2]
            if backup_traj.shape[0] > 0:
                ax.plot(
                    backup_traj[:, 0],
                    backup_traj[:, 1],
                    color=("red" if has_failed else "green"),
                    alpha=MONTE_CARLO_PLOT_CONFIG["trajectory_alpha"],
                    linewidth=MONTE_CARLO_PLOT_CONFIG["trajectory_linewidth"],
                )

        if has_failed:
            ax.scatter(
                traj[fail_idx, 0],
                traj[fail_idx, 1],
                marker="x",
                s=50,
                color="red",
                zorder=6,
            )

        if s < traj.shape[0] and s <= fail_idx:
            ax.scatter(
                traj[s, 0],
                traj[s, 1],
                s=30,
                color="yellow",
                edgecolors="black",
                linewidths=0.4,
                zorder=5,
            )

    # Calculate plot limits based on actual trajectories
    all_xy = plot_states[:, :, :2].reshape(-1, 2)
    padding = ANIMATION_PADDING
    x_min, x_max = all_xy[:, 0].min(), all_xy[:, 0].max()
    y_min, y_max = all_xy[:, 1].min(), all_xy[:, 1].max()

    # Ensure goal and obstacles are in frame
    x_min = min(x_min, float(target_pos_np[0]), float(np.min(obstacles_np[:, 0] - obstacles_np[:, 2])))
    x_max = max(x_max, float(target_pos_np[0]), float(np.max(obstacles_np[:, 0] + obstacles_np[:, 2])))
    y_min = min(y_min, float(target_pos_np[1]), float(np.min(obstacles_np[:, 1] - obstacles_np[:, 2])))
    y_max = max(y_max, float(target_pos_np[1]), float(np.max(obstacles_np[:, 1] + obstacles_np[:, 2])))

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim([x_min - padding, x_max + padding])
    ax.set_ylim([y_min - padding, y_max + padding])
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.grid(True, linestyle=":", alpha=MONTE_CARLO_PLOT_CONFIG["grid_alpha"])
    ax.set_title(f"Monte Carlo Rollouts (Switching Time = {s})")

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="blue", lw=2, label="Nominal Segment"),
        Line2D([0], [0], color="green", lw=2, label="Backup Segment (Safe)"),
        Line2D([0], [0], color="red", lw=2, label="Backup Segment (Crash)"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="yellow",
            markeredgecolor="black",
            markersize=8,
            label="Switching Point",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="w",
            markeredgecolor="red",
            markersize=8,
            label="Crash Point",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="gold",
            markeredgecolor="black",
            markersize=12,
            label="Goal",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="black",
            markersize=8,
            label="Start",
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
    )

    output_dir = Path(params.get("output_dir", monte_carlo_params["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / MONTE_CARLO_PLOT_CONFIG["save_filename"].format(s=s)
    plt.savefig(
        save_path,
        dpi=PLOT_DPI,
        bbox_inches=MONTE_CARLO_PLOT_CONFIG["bbox_inches"],
    )
    plt.close(fig)
    print(f"Saved plot to {save_path}")


def run_true_probability_monte_carlo(params=None, dataset=None, plot_override_s=None, verbose=True):
    if params is None:
        params = dict(monte_carlo_params)
    else:
        params = dict(params)

    seed = params.get("seed", monte_carlo_params["seed"])
    n_runs = params.get("n_runs", monte_carlo_params["n_runs"])
    horizon = params.get("horizon", monte_carlo_params["horizon"])
    dataset_size = params.get("dataset_size", monte_carlo_params["dataset_size"])
    switching_times = params.get("switching_times", monte_carlo_params["switching_times"])
    initial_state = params.get("initial_state", monte_carlo_params["initial_state"])
    true_beta = params.get("true_beta", monte_carlo_params["true_beta"])
    params.setdefault("output_dir", monte_carlo_params["output_dir"])

    if verbose:
        print("=== Starting True Probability of Failure Monte Carlo Analysis ===")
        print(f"Number of runs per switching time: {n_runs}")
        print(f"Trajectory Horizon: {horizon}")
        print(f"Evaluating switching times: {switching_times}")
        print(f"True Beta: {true_beta}\n")

    key = jax.random.PRNGKey(seed)
    key, data_key, dist_key = jax.random.split(key, 3)

    # Generate or reuse offline empirical dataset
    if dataset is None:
        if verbose:
            print("Generating offline empirical dataset...")
        dataset = generate_offline_dataset(data_key, dataset_size=dataset_size)

    # Prepare obstacles
    obstacles_np = format_obstacles(OBSTACLES)
    obstacle_centers = jnp.asarray(obstacles_np[:, :2])
    obstacle_radii = jnp.asarray(obstacles_np[:, 2])

    nominal_policy_fn = lambda s: nominal_policy(s, target_pos, target_v)

    # Sample true disturbance sequences once for all switching times (or independent per switching time)
    if verbose:
        print(f"Sampling {n_runs} true disturbance sequences...")
    t0_sample = time.time()
    _, true_sequences = sample_coupled_true_disturbance_sequences(
        dist_key,
        dataset,
        n_sequences=n_runs,
        horizon=horizon,
        beta=true_beta,
    )
    if verbose:
        print(f"Sampling completed in {time.time() - t0_sample:.2f}s.\n")

    results = {}

    for s in switching_times:
        t0_run = time.time()
        # Evaluate trajectories in parallel using JAX
        rollout_states = rollout_bundle(
            initial_state,
            int(s),
            true_sequences,
            nominal_policy_fn,
            obstacle_centers,
            obstacle_radii,
        )
        # Compute collisions
        collisions = batch_check_collisions(rollout_states, obstacles_np)
        num_failures = int(np.sum(collisions))
        prob_failure = num_failures / float(n_runs)
        elapsed = time.time() - t0_run

        results[s] = {
            "num_failures": num_failures,
            "prob_failure": prob_failure,
            "prob_safety": 1.0 - prob_failure,
            "runtime_sec": elapsed,
        }

        if verbose:
            print(
                f"Switching Time s={s:3d} | Failures: {num_failures:5d}/{n_runs} | "
                f"True P(Failure): {prob_failure:.4f} | True P(Safety): {1.0 - prob_failure:.4f} | "
                f"Time: {elapsed:.2f}s"
            )

        should_plot = False
        if plot_override_s is None:
            should_plot = True
        elif plot_override_s != -1 and int(s) == int(plot_override_s):
            should_plot = True

        if should_plot:
            if verbose:
                print("Generating plot...")
            plot_mc_rollouts(int(s), np.array(rollout_states), np.array(collisions), obstacles_np, params)

    if verbose:
        print("\n=== Summary of Results ===")
        print("Switching Time | True P(Failure) | True P(Safety)")
        print("-" * 45)
        for s, res in results.items():
            print(f"{s:14d} | {res['prob_failure']:15.4f} | {res['prob_safety']:14.4f}")

    return results


if __name__ == "__main__":
    run_true_probability_monte_carlo()
