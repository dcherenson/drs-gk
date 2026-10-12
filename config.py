"""Edit experiment settings here, then start a new run to apply them."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

# Dynamics, scenario, and controller settings
DT = 0.05

V_MIN = 9.0

V_MAX = 20.0

A_MIN = -5.0

A_MAX = 5.0

W_MAX = jnp.pi / 4.0

W_MIN = -W_MAX

DIST_DIM = 4  # [xi_a, xi_w, eta_a, eta_w]

CONTROL_DIM = 2  # [ua, uw]

TRUE_BETA = 0.01

K_P = 1.0

K_THETA = 2.0

SENSOR_NOISE_SCALE = 1.0  # Scale of the skew-normal sensor noise
SENSOR_NOISE_SHAPE = 5.0
SENSOR_NOISE_CLIP = 3.0

N_STEPS = 300

GATEKEEPER_VISUAL_RADIUS = 10000.0

OBSTACLES = np.array([[-10.0, 11.0, 15.0], [20.0, -20.0, 10.0]])

state_0 = jnp.array([-50.0, -10.0, 0.0, 10.0])

target_pos = jnp.array([50.0, 0.0])

target_v = 15.0

OPEN_LOOP = True

gk_params = {
    "M": 50,  # Number of candidate switching times
    "T": 75,  # Time horizon for rollouts
    "N": 1000,  # Number of noise samples per candidate
    "epsilon": 0.05,  # DRCCP violation threshold
    "delta": 0.1,  # Confidence level for statistical guarantees
    "sensor_noise_scale": SENSOR_NOISE_SCALE,  # Scale parameter (omega) for skew-normal noise
    "sensor_noise_shape": SENSOR_NOISE_SHAPE,  # Shape parameter (alpha) for skew-normal noise
    "beta": TRUE_BETA,  # Wasserstein radius
    "open_loop": OPEN_LOOP,
    "K_trials": 100,  # Number of independent trials for statistical confidence evaluation
}

cbf_params = {
    "T": gk_params["T"],  # True-process sequence horizon; not a CBF rollout parameter
    "alpha1": 2.0,  # Fixed in requested sweeps
    "epsilon": 0.1,  # DRCCP violation threshold
    "beta": TRUE_BETA,  # Wasserstein radius in CBF constraint
    "sample_count": 1000,  # Disturbance samples used by CBF optimization
}

pcis_params = {
    "alpha": 0.01,
    "num_samples": 1000,
    "progress_interval": 100,
}


@dataclass
class PolyhedralConstraints:
    H: np.ndarray
    h: np.ndarray


@dataclass
class WTMPCConfig:
    horizon: int
    n_samples: int
    dt: float
    w_max: np.ndarray

    gamma: float
    epsilon: float

    Q: np.ndarray
    R: np.ndarray
    Qf: np.ndarray

    q_lqr: np.ndarray
    r_lqr: np.ndarray

    state_constraints: PolyhedralConstraints
    input_constraints: PolyhedralConstraints


wt_mpc_params = WTMPCConfig(
    horizon=20,
    n_samples=20,
    gamma=0.1,
    epsilon=TRUE_BETA,
    dt=DT,
    Q=np.diag(
        [
            10.0,
            10.0,
            0.0,
            0.5,
        ]
    ),
    R=np.diag(
        [
            0.8,
            0.8,
        ]
    ),
    Qf=np.diag(
        [
            20.0,
            20.0,
            0.0,
            1.0,
        ]
    ),
    state_constraints=PolyhedralConstraints(
        H=np.array(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, -1.0],
            ]
        ),
        h=np.array(
            [
                V_MAX,
                -V_MIN,
            ]
        ),
    ),
    input_constraints=PolyhedralConstraints(
        H=np.array(
            [
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
            ]
        ),
        h=np.array(
            [
                A_MAX,
                -A_MIN,
                W_MAX,
                -W_MIN,
            ]
        ),
    ),
    # Support of the disturbance set:
    # W = {w : -w_max <= w <= w_max}
    w_max=np.array(
        [
            0.01,
            0.01,
            0.01,
            0.01,
        ]
    ),
    q_lqr=np.diag(
        [
            10.0,
            10.0,
            0.5,
            0.5,
        ]
    ),
    r_lqr=np.diag(
        [
            1.5,
            1.5,
        ]
    ),
)

# Common simulation and uncertainty settings
GOAL_TOLERANCE = 7.5
DEFAULT_GOAL_TOLERANCE = 5.0
WT_MPC_GOAL_TOLERANCE = 5.0
WT_MPC_SLOWDOWN_DISTANCE = 10.0
WT_MPC_OBSTACLE_MARGIN = 0.0
DATASET_SIZE = 5000
NOMINAL_NOISE_MEAN = np.array([0.0, 0.0, 0.0, 0.0])
NOMINAL_NOISE_STD = np.array([0.05, 0.02, 0.03, 0.03])
OUTLIER_NOISE_MEAN = np.array([-0.15, 0.08, -0.10, 0.05])
OUTLIER_NOISE_STD = np.array([0.02, 0.01, 0.05, 0.05])
OUTLIER_PROBABILITY = 0.2
PCIS_GRID_SIZE = (25, 25, 8, 4)
PCIS_GRID_MARGIN = 1.0
PCIS_GRID_PADDING = 2.0

# Runner and output settings
RUN_CONFIG = {
    "experiment": "compare",
    "seed": 420,
    "controllers": ["drs-gk", "gatekeeper", "smps", "dr-cbf", "wt-mpc"],
    "output_dir": "output",
    "animation": True,
    "verbose": False,
    "plot_filename": "comparison.png",
    "animation_filename": "comparison.gif",
    "metrics_filename": "metrics.csv",
}
PLOT_SEED = 42
ANIMATION_FPS = 10
ANIMATION_PADDING = 10.0
PLOT_DPI = 150
QUICK_CONFIG = {
    "controllers": ["drs-gk"],
    "steps": 600,
    "gk": {"M": 10, "T": 20, "N": 100, "K_trials": 3},
    "demo_gk": {"M": 40, "T": 75, "N": 300, "open_loop": False},
    "pcis": {"num_samples": 100, "progress_interval": 0},
    "monte_carlo": {"n_runs": 100, "horizon": 50, "n_plot_lines": 20},
    "sweep_n_seeds": 1,
    "initial_states": 3,
    "gk_sweeps": {"epsilon": [0.02, 0.05]},
    "epsilon_values": [0.02, 0.05],
    "beta_values": [0.01, 0.05],
}

# Experiment settings
EPSILON_SWEEP_VALUES = [0.001, 0.01, 0.05, 0.1, 0.5]

BETA_SWEEP_VALUES = [0.0, 0.001, 0.01, 0.1, 1.0]

INITIAL_STATE_COMPARISON_CONFIG = {
    "n_samples": 25,
    "seed": 42,
    "x_range": (-60.0, -30.0),
    "y_range": (-15.0, 15.0),
    "csv_filename": "initial_state_comparison.csv",
    "avoid_obstacles": False,
    "sensor_noise_scale": SENSOR_NOISE_SCALE,
}

HYPERPARAM_SWEEP_CONFIG = {
    "seed_start": 0,
    "n_seeds": 100,
    "gk_results_csv_path": "output/hyperparam_sweep_results_gk.csv",
    "gk_nominal": gk_params.copy(),
    "gk_sweeps": {
        # Explicit sweep values only.
        "M": [10, 20, 30, 40, 50],
        "T": [10, 20, 30, 40, 50],
        "N": [10, 100, 250, 500, 1000],
        "epsilon": EPSILON_SWEEP_VALUES,
        "delta": [0.001, 0.01, 0.05, 0.1, 0.5],
        "beta": BETA_SWEEP_VALUES,
    },
    "sensor_noise_scale": SENSOR_NOISE_SCALE,
}

EPS_BETA_SWEEP_CONFIG = {
    "seed_start": 0,
    "n_seeds": 25,
    "gk_results_csv_path": "output/eps_beta_sweep_gk.csv",
    # Initial state sampling
    "x_range": (-60.0, -30.0),
    "y_range": (-15.0, 15.0),
    "avoid_obstacles": True,
    # Sweep vectors
    "epsilon_values": [0.001, 0.01, 0.05, 0.1, 0.5],
    "beta_values": [0.01, 0.05, 0.1, 0.5, 1.0],
    "sensor_noise_scale": SENSOR_NOISE_SCALE,
}

monte_carlo_params = {
    "initial_state": state_0,
    "switching_times": [25, 30, 33, 35, 40],
    "n_runs": 10000,
    "horizon": 500,
    "dataset_size": DATASET_SIZE,
    "seed": 42,
    "true_beta": TRUE_BETA,
    "n_plot_lines": 100,
}

monte_carlo_params["output_dir"] = RUN_CONFIG["output_dir"]

# Sweep visualization settings
METRIC_SPECS = [("goal_reach_rate_pct", "Goal Reach Rate (%)"), ("avg_backup_pct", "Avg Backup Pct (%)")]
SHOW_ERROR_BARS = False
EXCLUDED_PLOT_PARAMS = {"M", "T"}
LOG_SCALE_PARAMS = {"N", "epsilon", "eps", "delta", "beta"}
PLOT_PARAM_LABELS = {"epsilon": "ε", "eps": "ε", "delta": "δ", "beta": "β", "alpha": "α", "alpha1": "α1"}

MONTE_CARLO_PLOT_CONFIG = {
    "figsize": (10, 6),
    "trajectory_alpha": 0.15,
    "trajectory_linewidth": 1,
    "grid_alpha": 0.3,
    "save_filename": "mc_rollouts_s_{s}.png",
    "bbox_inches": "tight",
}
CONTROLLER_STYLES = {
    "drs-gk": ("DRS-GK", "tab:blue"),
    "gatekeeper": ("Gatekeeper", "tab:red"),
    "smps": ("SMPS", "tab:green"),
    "dr-cbf": ("DR-CBF", "tab:purple"),
    "wt-mpc": ("WT-MPC", "tab:orange"),
}
COMPARISON_PLOT_CONFIG = {
    "figsize": (10, 6),
    "grid_alpha": 0.5,
    "trajectory_linewidth": 2,
    "start_markersize": 5,
    "goal_markersize": 12,
    "position_markersize": 5,
    "failure_markersize": 8,
    "legend_loc": "upper left",
}
OBSTACLE_STYLE = {"color": "red", "alpha": 0.3}

HYPERPARAM_PLOT_CONFIG = {
    "svg_fonttype": "none",
    "font_family": "DejaVu Sans",
    "n_cols": 2,
    "n_rows": 2,
    "subplot_width": 6,
    "subplot_height": 3.5,
    "colors": ["tab:blue", "tab:green", "tab:orange", "tab:red"],
    "markers": ["o", "s", "^", "D"],
    "label_fontsize": 20,
    "tick_fontsize": 20,
    "legend_fontsize": 20,
    "line_width": 2,
    "errorbar_line_width": 1.4,
    "errorbar_capsize": 4,
    "y_label": "%",
    "y_limits": (-5, 105),
    "y_ticks": [0, 25, 50, 75, 100],
    "x_tick_rotation": 45,
    "grid_alpha": 0.3,
    "legend_loc": "upper center",
    "legend_ncol": 2,
    "legend_frameon": False,
    "legend_bbox_to_anchor": (0.5, 0.99),
    "tight_layout_rect": (0.0, 0.0, 1.0, 0.95),
    "bbox_inches": "tight",
}
