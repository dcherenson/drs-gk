import numpy as np
import jax
import platform
from config import (
    A_MAX,
    A_MIN,
    DT,
    PCIS_GRID_MARGIN,
    PCIS_GRID_PADDING,
    PCIS_GRID_SIZE,
    TRUE_BETA,
    V_MIN,
    W_MAX,
    pcis_params,
)
from helper import (
    sample_true_disturbances,
    backup_policy,
)

# Force CPU backend if using JAX on Mac
if platform.system() == "Darwin":
    jax.config.update("jax_platform_name", "cpu")

V_MAX = float(V_MIN)


def dynamics_step(state, control, disturbance):
    px, py, theta, v = state
    ua, uw = control
    xi_a, xi_w, eta_a, eta_w = disturbance

    px_dot = v * np.cos(theta)
    py_dot = v * np.sin(theta)
    theta_dot = (1.0 + eta_w) * uw + (v * xi_w)
    v_dot = (1.0 + eta_a) * ua + (v * xi_a)

    next_px = px + (px_dot * DT)
    next_py = py + (py_dot * DT)
    next_theta = theta + (theta_dot * DT)
    next_theta = (next_theta + np.pi) % (2 * np.pi) - np.pi
    next_v = np.clip(v + (v_dot * DT), V_MIN, V_MAX)

    return np.array([next_px, next_py, next_theta, next_v])


def define_orbit_tube():
    # Grid definition (px, py, theta, v), matched to the DRS-GK dynamics.
    # The backup orbit radius is v / W_MAX, so cover the full speed range
    # with a bounding margin derived from maximum 1-step travel distance.
    max_orbit_radius = V_MAX / W_MAX
    grid_margin = V_MAX * DT + PCIS_GRID_MARGIN  # Tight margin to act as strict tracking constraints
    grid_extent = np.ceil(max_orbit_radius + grid_margin + PCIS_GRID_PADDING)
    px_vals = np.linspace(-grid_extent, grid_extent, PCIS_GRID_SIZE[0])
    py_vals = np.linspace(-grid_extent, grid_extent, PCIS_GRID_SIZE[1])
    theta_vals = np.linspace(-np.pi, np.pi, PCIS_GRID_SIZE[2] + 1)[:-1]
    v_vals = np.linspace(V_MIN, V_MAX, PCIS_GRID_SIZE[3])

    # The candidate set Q contains positions near the circular backup orbits over the
    # configured speed range. The PCIS algorithm will refine this set.
    min_orbit_radius = V_MIN / W_MAX
    states_Q = []
    for px in px_vals:
        for py in py_vals:
            r = np.sqrt(px**2 + py**2)
            if min_orbit_radius - grid_margin <= r <= max_orbit_radius + grid_margin:
                for theta in theta_vals:
                    for v in v_vals:
                        states_Q.append(np.array([px, py, theta, v]))
    return np.array(states_Q), (px_vals, py_vals, theta_vals, v_vals)


def get_nearest_grid_index(state, grid_axes):
    px_vals, py_vals, theta_vals, v_vals = grid_axes
    idx_px = np.argmin(np.abs(px_vals - state[0]))
    idx_py = np.argmin(np.abs(py_vals - state[1]))

    # Handle angle wrap-around for nearest neighbor
    d_theta = np.abs(theta_vals - state[2])
    d_theta = np.minimum(d_theta, 2 * np.pi - d_theta)
    idx_theta = np.argmin(d_theta)

    idx_v = np.argmin(np.abs(v_vals - state[3]))
    return (idx_px, idx_py, idx_theta, idx_v)


def get_nearest_grid_indices(states, grid_axes):
    """Vectorized nearest-grid lookup for an array of states."""
    px_vals, py_vals, theta_vals, v_vals = grid_axes
    states = np.asarray(states)

    idx_px = np.argmin(np.abs(px_vals[None, :] - states[:, 0, None]), axis=1)
    idx_py = np.argmin(np.abs(py_vals[None, :] - states[:, 1, None]), axis=1)

    d_theta = np.abs(theta_vals[None, :] - states[:, 2, None])
    d_theta = np.minimum(d_theta, 2 * np.pi - d_theta)
    idx_theta = np.argmin(d_theta, axis=1)

    idx_v = np.argmin(np.abs(v_vals[None, :] - states[:, 3, None]), axis=1)
    return idx_px, idx_py, idx_theta, idx_v


def dynamics_step_batch(state, control, disturbances):
    """Vectorized version of dynamics_step for one state and many noises."""
    state = np.asarray(state, dtype=float)
    control = np.asarray(control, dtype=float)
    disturbances = np.asarray(disturbances, dtype=float)

    px, py, theta, v = state
    ua, uw = control
    xi_a, xi_w, eta_a, eta_w = disturbances.T

    px_dot = v * np.cos(theta)
    py_dot = v * np.sin(theta)
    theta_dot = (1.0 + eta_w) * uw + (v * xi_w)
    v_dot = (1.0 + eta_a) * ua + (v * xi_a)

    next_px = np.full(xi_a.shape, px + (px_dot * DT))
    next_py = np.full(xi_a.shape, py + (py_dot * DT))
    next_theta = theta + (theta_dot * DT)
    next_theta = (next_theta + np.pi) % (2 * np.pi) - np.pi
    next_v = np.clip(v + (v_dot * DT), V_MIN, V_MAX)

    return np.column_stack((next_px, next_py, next_theta, next_v))


def compute_rcis(states_Q, grid_axes, controls_U, dist_boundary):
    # Initialize Qf indicator as a boolean mask over the state list
    # For fast nearest-neighbor lookups, a full grid mask is maintained
    grid_shape = tuple(len(ax) for ax in grid_axes)
    Qf_grid = np.zeros(grid_shape, dtype=bool)

    # Map valid initial states into the grid mask
    for s in states_Q:
        idx = get_nearest_grid_index(s, grid_axes)
        Qf_grid[idx] = True

    converged = False
    iteration = 0

    while not converged:
        iteration += 1
        next_Qf_grid = np.zeros_like(Qf_grid)
        changes = 0

        # Iterate over all states currently in Qf
        it = np.nditer(Qf_grid, flags=["multi_index"])
        for in_qf in it:
            if not in_qf:
                continue

            idx = it.multi_index
            state = np.array([grid_axes[0][idx[0]], grid_axes[1][idx[1]], grid_axes[2][idx[2]], grid_axes[3][idx[3]]])

            state_robust = False
            # Closed-loop policy evaluation
            u = np.asarray(backup_policy(state, np.array([0.0, 0.0])))
            u_robust = True
            for w in dist_boundary:
                next_state = dynamics_step(state, u, w)
                next_idx = get_nearest_grid_index(next_state, grid_axes)
                if not Qf_grid[next_idx]:
                    u_robust = False
                    break

            if u_robust:
                state_robust = True

            if state_robust:
                next_Qf_grid[idx] = True
            else:
                changes += 1

        if changes == 0:
            converged = True
        Qf_grid = next_Qf_grid

    # Extract RCIS states
    Qf_states = []
    it = np.nditer(Qf_grid, flags=["multi_index"])
    for in_qf in it:
        if in_qf:
            idx = it.multi_index
            Qf_states.append(
                np.array([grid_axes[0][idx[0]], grid_axes[1][idx[1]], grid_axes[2][idx[2]], grid_axes[3][idx[3]]])
            )

    return np.array(Qf_states), Qf_grid


def compute_pcis(
    states_Q,
    Qf_grid,
    grid_axes,
    controls_U,
    epsilon,
    num_samples,
    key=None,
    dataset=None,
    progress_interval=100,
    true_beta=None,
):
    if true_beta is None:
        true_beta = TRUE_BETA

    if key is None:
        raise ValueError("compute_pcis requires a JAX PRNG key when sampling disturbances.")
    if dataset is None:
        raise ValueError("compute_pcis requires the nominal offline dataset.")

    num_samples = int(num_samples)
    if num_samples < 1:
        raise ValueError("num_samples must be positive.")

    # The disturbance law is independent of the candidate state. Reusing one
    # Monte Carlo sample set across states is common-random-number sampling:
    # it preserves the estimator for each state while avoiding one JAX call
    # per state.
    disturbances = np.asarray(
        sample_true_disturbances(key, dataset, num_samples, true_beta),
        dtype=float,
    )
    if disturbances.ndim == 1:
        disturbances = disturbances.reshape(1, -1)

    states_Q = np.asarray(states_Q)
    pcis_states = []
    valid_controls_map = {}
    total_states = len(states_Q)
    progress_interval = int(progress_interval) if progress_interval is not None else 0

    print(
        f"PCIS: evaluating {total_states} states with {len(controls_U)} controls and {num_samples} disturbances...",
        flush=True,
    )

    for state_index, state in enumerate(states_Q):
        state_valid_controls = []

        # Closed-loop policy evaluation
        u = np.asarray(backup_policy(state, np.array([0.0, 0.0])))
        next_states = dynamics_step_batch(state, u, disturbances)
        idx_px, idx_py, idx_theta, idx_v = get_nearest_grid_indices(next_states, grid_axes)
        success_count = np.count_nonzero(Qf_grid[idx_px, idx_py, idx_theta, idx_v])

        prob = success_count / num_samples
        if prob >= epsilon:
            state_valid_controls.append(u)

        if len(state_valid_controls) > 0:
            pcis_states.append(state)
            valid_controls_map[tuple(state)] = state_valid_controls

        completed = state_index + 1
        if progress_interval > 0 and (completed % progress_interval == 0 or completed == total_states):
            print(f"PCIS progress: {completed}/{total_states} states", flush=True)

    return np.array(pcis_states), valid_controls_map


def build_pcis_data(
    key,
    dataset,
    alpha=pcis_params["alpha"],
    num_samples=pcis_params["num_samples"],
    progress_interval=pcis_params["progress_interval"],
    true_beta=None,
):
    """Compute PCIS once and package the arrays needed by DRS-GK."""
    epsilon = 1.0 - alpha
    controls_U = [np.array([a, w]) for a in [A_MIN, 0.0, A_MAX] for w in [-W_MAX, 0.0, W_MAX]]
    dist_min = np.min(dataset, axis=0)
    dist_max = np.max(dataset, axis=0)
    dist_boundary = [
        np.array([float(x1), float(x2), float(x3), float(x4)])
        for x1 in [dist_min[0], dist_max[0]]
        for x2 in [dist_min[1], dist_max[1]]
        for x3 in [dist_min[2], dist_max[2]]
        for x4 in [dist_min[3], dist_max[3]]
    ]

    states_Q, grid_axes = define_orbit_tube()
    Qf_states, Qf_grid = compute_rcis(states_Q, grid_axes, controls_U, dist_boundary)
    pcis_states, valid_controls_map = compute_pcis(
        states_Q,
        Qf_grid,
        grid_axes,
        controls_U,
        epsilon,
        num_samples,
        key=key,
        dataset=dataset,
        progress_interval=progress_interval,
        true_beta=true_beta,
    )
    if len(pcis_states) == 0:
        raise ValueError("The computed PCIS is empty; DRS-GK cannot use it as a terminal set.")

    px_vals, py_vals, theta_vals, v_vals = grid_axes
    theta_step = 2.0 * np.pi / len(theta_vals)
    tolerances = np.array(
        [
            0.5 * abs(px_vals[1] - px_vals[0]),
            0.5 * abs(py_vals[1] - py_vals[0]),
            0.5 * theta_step,
            0.5 * abs(v_vals[1] - v_vals[0]),
        ]
    )

    # Only 1 closed-loop action per state
    control_table = np.zeros((len(pcis_states), 1, 2), dtype=float)
    control_mask = np.zeros((len(pcis_states), 1), dtype=bool)
    for state_index, state in enumerate(pcis_states):
        if valid_controls_map[tuple(state)]:
            control = valid_controls_map[tuple(state)][0]
            control_table[state_index, 0] = control
            control_mask[state_index, 0] = True

    pcis_grid = np.zeros_like(Qf_grid, dtype=bool)
    pcis_control_grid = np.zeros(Qf_grid.shape + (1, 2), dtype=float)
    pcis_control_mask_grid = np.zeros(Qf_grid.shape + (1,), dtype=bool)
    for state_index, state in enumerate(pcis_states):
        grid_index = get_nearest_grid_index(state, grid_axes)
        pcis_grid[grid_index] = True
        pcis_control_grid[grid_index] = control_table[state_index]
        pcis_control_mask_grid[grid_index] = control_mask[state_index]

    return {
        "states": np.asarray(pcis_states, dtype=float),
        "tolerances": tolerances,
        "controls": control_table,
        "control_mask": control_mask,
        "grid": pcis_grid,
        "control_grid": pcis_control_grid,
        "control_mask_grid": pcis_control_mask_grid,
        "position_radius": float(np.max(np.linalg.norm(pcis_states[:, :2], axis=1))),
        "inner_radius": float(np.min(np.linalg.norm(pcis_states[:, :2], axis=1))),
        "torus_width": float(np.max(np.abs(np.linalg.norm(pcis_states[:, :2], axis=1) - pcis_states[:, 3] / W_MAX))),
        "states_Q": states_Q,
        "Qf_states": Qf_states,
        "Qf_grid": Qf_grid,
        "grid_axes": grid_axes,
        "valid_controls": valid_controls_map,
        "alpha": alpha,
    }
