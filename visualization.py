"""Plots and GIFs for controller comparisons."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

from config import ANIMATION_FPS, ANIMATION_PADDING, COMPARISON_PLOT_CONFIG, CONTROLLER_STYLES, PLOT_DPI
from helper import add_obstacles_to_axis, format_obstacles


def _setup_axis(ax, results, obstacles, goal):
    add_obstacles_to_axis(ax, obstacles)
    trajectories = [np.asarray(result["states"])[:, :2] for result in results.values()]
    points = np.concatenate(trajectories + [np.asarray(goal)[None, :]])
    obstacles = format_obstacles(obstacles)
    lower = np.minimum(points.min(axis=0), (obstacles[:, :2] - obstacles[:, 2, None]).min(axis=0))
    upper = np.maximum(points.max(axis=0), (obstacles[:, :2] + obstacles[:, 2, None]).max(axis=0))
    ax.set_xlim(lower[0] - ANIMATION_PADDING, upper[0] + ANIMATION_PADDING)
    ax.set_ylim(lower[1] - ANIMATION_PADDING, upper[1] + ANIMATION_PADDING)
    start = trajectories[0][0]
    ax.plot(*start, "ks", markersize=COMPARISON_PLOT_CONFIG["start_markersize"], label="Start")
    ax.plot(
        *goal,
        marker="*",
        color="gold",
        markeredgecolor="black",
        markersize=COMPARISON_PLOT_CONFIG["goal_markersize"],
        label="Goal",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.grid(True, linestyle=":", alpha=COMPARISON_PLOT_CONFIG["grid_alpha"])


def plot_comparison(results, obstacles, goal, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=COMPARISON_PLOT_CONFIG["figsize"])
    _setup_axis(ax, results, obstacles, goal)
    for name, result in results.items():
        label, color = CONTROLLER_STYLES[name]
        states = np.asarray(result["states"])
        ax.plot(
            states[:, 0],
            states[:, 1],
            color=color,
            linewidth=COMPARISON_PLOT_CONFIG["trajectory_linewidth"],
            label=label,
        )
        if result["stats"].get("infeasible") or result["stats"].get("hit_obstacle"):
            ax.plot(*states[-1, :2], marker="x", color=color, markersize=COMPARISON_PLOT_CONFIG["failure_markersize"])
    ax.legend(loc=COMPARISON_PLOT_CONFIG["legend_loc"])
    fig.tight_layout()
    fig.savefig(save_path, dpi=PLOT_DPI)
    plt.close(fig)
    return save_path


def animate_comparison(results, obstacles, goal, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=COMPARISON_PLOT_CONFIG["figsize"])
    _setup_axis(ax, results, obstacles, goal)
    artists = []
    for name, result in results.items():
        label, color = CONTROLLER_STYLES[name]
        (line,) = ax.plot([], [], color=color, linewidth=COMPARISON_PLOT_CONFIG["trajectory_linewidth"], label=label)
        (point,) = ax.plot([], [], marker="o", color=color, markersize=COMPARISON_PLOT_CONFIG["position_markersize"])
        artists.append((np.asarray(result["states"]), line, point))
    ax.legend(loc=COMPARISON_PLOT_CONFIG["legend_loc"])
    fig.tight_layout()

    def update(frame):
        for states, line, point in artists:
            index = min(frame, len(states) - 1)
            line.set_data(states[: index + 1, 0], states[: index + 1, 1])
            point.set_data([states[index, 0]], [states[index, 1]])
        return [artist for _, line, point in artists for artist in (line, point)]

    movie = animation.FuncAnimation(
        fig,
        update,
        frames=max(len(states) for states, _, _ in artists),
        interval=1000 / ANIMATION_FPS,
        blit=False,
    )
    movie.save(save_path, writer=animation.PillowWriter(fps=ANIMATION_FPS))
    plt.close(fig)
    return save_path
