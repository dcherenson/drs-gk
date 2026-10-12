import time
import jax
import jax.numpy as jnp
import numpy as np

from config import (
    CONTROL_DIM,
    DEFAULT_GOAL_TOLERANCE,
    DT,
    DIST_DIM,
    N_STEPS,
    SENSOR_NOISE_CLIP,
    SENSOR_NOISE_SHAPE,
    TRUE_BETA,
    W_MAX,
    gk_params,
    target_pos,
    target_v,
)
from helper import (
    backup_policy,
    closest_obstacle_center_jax,
    closest_obstacle_index,
    collides_with_any_obstacle,
    dynamics_step,
    format_obstacles,
    generate_skew_normal_noise,
    nominal_policy,
    sample_coupled_true_process_sequence,
    visible_obstacle_indices,
)


@jax.jit(static_argnames=["nominal_policy_fn"])
def compute_H_vanilla(state, s, nominal_policy_fn, obstacle_centers, obstacle_radii):
    """
    Rolls out the dynamics and computes the safety margin H_T and its gradient.
    s: Scalar switching time index (0 to T-1).
    W: Noise trajectory of shape (T, DIST_DIM).
    """

    def scan_step(carry, inputs):
        current_state = carry
        t = inputs
        backup_center = closest_obstacle_center_jax(current_state, obstacle_centers, obstacle_radii)

        # Select policy based on switching time s
        u = jax.lax.cond(
            t < s,
            lambda _: nominal_policy_fn(current_state),
            lambda _: backup_policy(current_state, backup_center),
            operand=None,
        )
        next_state = dynamics_step(current_state, u, jnp.zeros(DIST_DIM))

        distances = jnp.linalg.norm(obstacle_centers - next_state[:2], axis=1)
        margin = jnp.min(distances - obstacle_radii)

        return next_state, margin

    # Create time vector for the scan loop
    T = gk_params["T"]
    time_indices = jnp.arange(T)

    # Execute rollout
    final_state, margins = jax.lax.scan(scan_step, state, time_indices)

    # H_T is the minimum safety margin over the horizon
    # 1. Minimum safety margin during the finite horizon
    H_traj = jnp.min(margins)

    # 2. Terminal PCIS margin (h_c)
    px_T, py_T, theta_T, v_T = final_state[0], final_state[1], final_state[2], final_state[3]

    # Calculate the center of the loiter circle initiated at the terminal state
    terminal_backup_center = closest_obstacle_center_jax(final_state, obstacle_centers, obstacle_radii)
    rel_px_T = px_T - terminal_backup_center[0]
    rel_py_T = py_T - terminal_backup_center[1]
    turn_metric = rel_py_T * jnp.cos(theta_T) - rel_px_T * jnp.sin(theta_T)
    S_w = jnp.sign(turn_metric + 1e-6)

    # Radius of the turn at the terminal velocity
    R_turn = v_T / W_MAX

    # Coordinates of the loiter center
    center_x = px_T - S_w * R_turn * jnp.sin(theta_T)
    center_y = py_T + S_w * R_turn * jnp.cos(theta_T)

    loiter_center = jnp.array([center_x, center_y])
    dist_to_centers = jnp.linalg.norm(obstacle_centers - loiter_center, axis=1)

    # The minimum distance between the resulting loiter circle and all active obstacles
    H_terminal = jnp.min(dist_to_centers - R_turn - obstacle_radii)

    # The final safety function is the minimum of the trajectory margin and terminal margin
    H_T = jnp.minimum(H_traj, H_terminal)
    return H_T


def execute_vanilla_gatekeeper(state, params, nominal_policy_fn, active_obstacles, prev_switching_time_offset):
    """
    Evaluates candidates in parallel using JAX, then computes the statistical
    bounds in Python using SciPy to select the optimal switching time.
    """
    M = params["M"]

    active_obstacles_np = format_obstacles(active_obstacles, allow_empty=True)
    obstacle_centers = jnp.asarray(active_obstacles_np[:, :2])
    obstacle_radii = jnp.asarray(active_obstacles_np[:, 2])
    # 2. Define candidate switching times [0, 1, ..., M-1] and Evaluate all candidates in JAX
    # Returns a JAX array of shape (M,) indicating whether each candidate is safe (True) or unsafe (False)
    jax_safe = jax.vmap(compute_H_vanilla, in_axes=(None, 0, None, None, None))(
        state,
        jnp.arange(M),
        nominal_policy_fn,
        obstacle_centers,
        obstacle_radii,
    )

    safe_candidates = jax_safe >= 0
    # choose the largest safe candidate (if any)
    best_s_relative = 0
    if safe_candidates.sum() > 0:
        best_s_relative = jnp.where(safe_candidates == 1)[0][-1]
    else:
        best_s_relative = prev_switching_time_offset - 1
    return best_s_relative


def run_vanilla_gatekeeper_simulation(
    env_key,
    initial_state,
    dataset,
    params,
    obstacles,
    gatekeeper_visual_radius,
    verbose=True,
    return_stats=False,
):
    obstacles_np = format_obstacles(obstacles)
    state = initial_state
    state_history = [state]
    control_history = []
    rollout_history = []
    switch_offset_history = []
    time_history = []
    backup_steps = 0
    total_steps = 0
    hit_obstacle = False
    goal_reached = False
    time_to_goal = np.inf

    nominal_policy_fn = lambda s: nominal_policy(s, target_pos, target_v)

    true_sequence_horizon = int(params.get("T", gk_params.get("T", N_STEPS)))
    true_shift_direction = params.get("true_shift_direction", None)
    true_sequence_metric = params.get("true_sequence_metric", "l2")

    _, true_disturbance_sequence = sample_coupled_true_process_sequence(
        env_key,
        dataset,
        n_steps=N_STEPS,
        sequence_horizon=true_sequence_horizon,
        beta=TRUE_BETA,
        shift_direction=true_shift_direction,
        sequence_metric=true_sequence_metric,
    )

    switching_time = 0  # Initialize switching time to 0 (start with nominal policy)
    for t in range(N_STEPS):
        # Split the keys independently
        env_key, _, sensor_key = jax.random.split(env_key, 3)
        t0 = time.time()

        # Apply sensor noise to obstacles
        scale = params.get("sensor_noise_scale", 0.0)
        shape_val = params.get("sensor_noise_shape", SENSOR_NOISE_SHAPE)
        if scale > 0:
            sensor_noise = generate_skew_normal_noise(sensor_key, obstacles_np.shape, scale, shape_val)
            sensor_noise = np.clip(
                sensor_noise,
                -scale * SENSOR_NOISE_CLIP,
                scale * SENSOR_NOISE_CLIP,
            )
            sensed_obstacles = obstacles_np + sensor_noise
        else:
            sensed_obstacles = obstacles_np

        # 1. Gatekeeper safety check
        active_obstacle_indices = visible_obstacle_indices(state, sensed_obstacles, gatekeeper_visual_radius)
        active_obstacles = sensed_obstacles[active_obstacle_indices]
        prev_switching_time_offset = switching_time
        switch_offset = execute_vanilla_gatekeeper(
            state, params, nominal_policy_fn, active_obstacles, prev_switching_time_offset
        )

        switching_time = t + switch_offset
        t1 = time.time() - t0
        time_history.append(t1)
        if verbose:
            print(
                f"Step {t}: Selected switching time = {switching_time}, Vanilla Gatekeeper computation time = {t1:.3f} seconds"
            )

        # 2. Policy selection
        if t < switching_time:
            u_applied = nominal_policy_fn(state)
        else:
            closest_idx = closest_obstacle_index(state, sensed_obstacles)
            u_applied = backup_policy(state, jnp.asarray(sensed_obstacles[closest_idx, :2]))
            backup_steps += 1

        # sample true disturbance
        W = true_disturbance_sequence[t]
        state = dynamics_step(state, u_applied, W)

        # Log data
        state_history.append(state)
        control_history.append(u_applied)
        total_steps += 1

        # check collision
        if collides_with_any_obstacle(state, obstacles_np):
            hit_obstacle = True
            if verbose:
                print(f"Collision detected at step {t} with state {state[:2]}")
            break

        # check if goal is reached
        if jnp.linalg.norm(state[:2] - target_pos) <= DEFAULT_GOAL_TOLERANCE:
            goal_reached = True
            time_to_goal = float((t + 1) * DT)
            if verbose:
                print(f"Goal reached at step {t} with state {state[:2]}")
            break

    controls_array = (
        jnp.stack(control_history) if control_history else jnp.empty((0, CONTROL_DIM), dtype=state_history[0].dtype)
    )
    result = (
        jnp.stack(state_history),
        controls_array,
        rollout_history,
        switch_offset_history,
    )
    if not return_stats:
        return result

    backup_usage_pct = 100.0 * (backup_steps / total_steps) if total_steps > 0 else 0.0
    stats = {
        "backup_usage_pct": float(backup_usage_pct),
        "hit_obstacle": bool(hit_obstacle),
        "time_to_goal": float(time_to_goal) if goal_reached else float(np.inf),
        "mean_computation_time": float(np.mean(time_history)) if time_history else 0.0,
    }
    return (*result, stats)
