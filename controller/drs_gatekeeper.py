import time
import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats

from config import (
    CONTROL_DIM,
    DEFAULT_GOAL_TOLERANCE,
    DIST_DIM,
    DT,
    N_STEPS,
    SENSOR_NOISE_CLIP,
    SENSOR_NOISE_SHAPE,
    TRUE_BETA,
    W_MAX,
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
    sample_empirical_disturbances,
    visible_obstacle_indices,
)


def orbit_center_for_state(state, obstacle_centers, obstacle_radii):
    """Return the center of the backup orbit associated with a state."""
    backup_center = closest_obstacle_center_jax(state, obstacle_centers, obstacle_radii)
    rel_px = state[0] - backup_center[0]
    rel_py = state[1] - backup_center[1]
    turn_metric = rel_py * jnp.cos(state[2]) - rel_px * jnp.sin(state[2])
    turn_direction = jnp.sign(turn_metric + 1e-6)
    turn_radius = state[3] / W_MAX
    return jnp.array(
        [
            state[0] - turn_direction * turn_radius * jnp.sin(state[2]),
            state[1] + turn_direction * turn_radius * jnp.cos(state[2]),
        ]
    )


def pcis_grid_obstacle_clearance(state, orbit_center, pcis_inputs):
    """Conservative clearance of the translated PCIS annulus from obstacles based on current speed."""
    v = state[3]
    torus_width = pcis_inputs["torus_width"]
    dynamic_position_radius = (v / W_MAX) + torus_width
    distances = jnp.linalg.norm(pcis_inputs["obstacle_centers"] - orbit_center[None, :], axis=1)
    outside_margin = distances - pcis_inputs["obstacle_radii"] - dynamic_position_radius
    inside_margin = pcis_inputs["inner_radius"] - (distances + pcis_inputs["obstacle_radii"])
    center_clearances = jnp.maximum(outside_margin, inside_margin)
    return jnp.min(center_clearances)


def pcis_grid_terminal_margin(
    final_state,
    orbit_obstacle_centers,
    orbit_obstacle_radii,
    pcis_inputs,
):
    orbit_center = orbit_center_for_state(
        final_state,
        orbit_obstacle_centers,
        orbit_obstacle_radii,
    )
    obstacle_margin = pcis_grid_obstacle_clearance(
        final_state,
        orbit_center,
        pcis_inputs,
    )
    return obstacle_margin


@jax.jit(static_argnames=["nominal_policy_fn"])
def compute_H_and_grad(
    state,
    s,
    W_flat,
    nominal_policy_fn,
    obstacle_centers,
    obstacle_radii,
    pcis_inputs,
):
    """
    Rolls out the dynamics and computes the safety margin H_T and its gradient.
    s: Scalar switching time index (0 to T-1).
    W_flat: Flat noise vector of shape (T * DIST_DIM + O * 3,).
    """
    num_obs = obstacle_centers.shape[0]
    obs_noise_dim = num_obs * 3
    if obs_noise_dim > 0:
        W_process_flat = W_flat[:-obs_noise_dim]
        W_obs_flat = W_flat[-obs_noise_dim:]
        W = W_process_flat.reshape((-1, DIST_DIM))
        obs_noise = W_obs_flat.reshape((num_obs, 3))
        noisy_centers = obstacle_centers + obs_noise[:, :2]
        noisy_radii = obstacle_radii + obs_noise[:, 2]
    else:
        W = W_flat.reshape((-1, DIST_DIM))
        noisy_centers = obstacle_centers
        noisy_radii = obstacle_radii

    def scan_step(carry, inputs):
        current_state = carry
        t, w_t = inputs
        backup_center = closest_obstacle_center_jax(current_state, noisy_centers, noisy_radii)

        # Select policy based on switching time s
        u = jax.lax.cond(
            t < s,
            lambda _: nominal_policy_fn(current_state),
            lambda _: backup_policy(
                current_state,
                backup_center,
            ),
            operand=None,
        )

        next_state = dynamics_step(current_state, u, w_t)

        distances = jnp.linalg.norm(noisy_centers - next_state[:2], axis=1)
        margin = jnp.min(distances - noisy_radii)

        return next_state, margin

    # Create time vector for the scan loop
    T = W.shape[0]
    time_indices = jnp.arange(T)
    scan_inputs = (time_indices, W)

    # Execute rollout
    final_state, margins = jax.lax.scan(scan_step, state, scan_inputs)

    # H_T is the minimum safety margin over the horizon
    # 1. Minimum safety margin during the finite horizon
    H_traj = jnp.min(margins)

    # 2. Require entry into the translated PCIS and ensure the entire
    # translated PCIS remains clear of every obstacle.
    H_terminal = pcis_grid_terminal_margin(
        final_state,
        noisy_centers,
        noisy_radii,
        pcis_inputs,
    )

    # The final safety function is the minimum of the trajectory margin and terminal margin
    H_T = jnp.minimum(H_traj, H_terminal)
    return H_T


# Create a function that returns both the value H_T and the gradient d(H_T)/dW
# argnums=2 specifies taking the derivative w.r.t the noise trajectory W
get_H_and_grad = jax.value_and_grad(compute_H_and_grad, argnums=2)


@jax.jit(static_argnames=["nominal_policy_fn", "beta", "p_norm"])
def evaluate_candidate(
    state,
    s,
    W_samples,
    nominal_policy_fn,
    obstacle_centers,
    obstacle_radii,
    pcis_inputs,
    beta,
    p_norm=1.0,
):
    """
    Evaluates a single switching time s across N noise trajectories.
    W_samples: Array of shape (N, T * DIST_DIM + O * 3).
    """
    # Vectorize the rollout and gradient over the N dimension
    # W_samples has shape (N, T * DIST_DIM + O * 3) -> axis 0
    # obstacle_centers has shape (O, 2) -> None
    # obstacle_radii has shape (O,) -> None
    vmap_eval = jax.vmap(
        get_H_and_grad,
        in_axes=(None, None, 0, None, None, None, None),
    )

    H_values, grad_W = vmap_eval(
        state,
        s,
        W_samples,
        nominal_policy_fn,
        obstacle_centers,
        obstacle_radii,
        pcis_inputs,
    )

    # Compute L_H: Maximum norm of the gradient across all N samples
    # grad_W has shape (N, T * DIST_DIM + O * 3). Norm is taken over the flat noise dimension.
    grad_norms = jax.vmap(jnp.linalg.norm)(grad_W)
    L_H = jnp.max(grad_norms)

    # Determine the robustness buffer and count failures
    buffer = L_H * (beta ** (1.0 / p_norm))
    failure_mask = H_values <= buffer
    failures = jnp.sum(failure_mask)

    return failures, L_H, H_values, failure_mask


@jax.jit(static_argnames=["nominal_policy_fn", "beta"])
def evaluate_all_candidates(
    state,
    s_array,
    W_samples,
    nominal_policy_fn,
    obstacle_centers,
    obstacle_radii,
    pcis_inputs,
    beta,
):
    """
    Evaluates all M candidate switching times in parallel.
    s_array: Array of candidate times, shape (M,).
    W_samples: Array of shape (M, N, W_dim).
    """
    # Vectorize over the M dimension (mapping s_array at 0 and W_samples at 0)
    vmap_candidates = jax.vmap(
        evaluate_candidate,
        in_axes=(None, 0, 0, None, None, None, None, None),
    )

    candidate_failures, L_H_array, H_values_array, failure_masks = vmap_candidates(
        state,
        s_array,
        W_samples,
        nominal_policy_fn,
        obstacle_centers,
        obstacle_radii,
        pcis_inputs,
        beta,
    )
    return candidate_failures, L_H_array, H_values_array, failure_masks


def compute_q_bar(k_array, N, delta):
    """
    Computes the Clopper-Pearson upper bound for an array of failure counts.
    Mathematically equivalent to the maximum q satisfying Binomial(k; N, q) >= delta.
    """
    # Vectorized Beta distribution percent point function (inverse CDF)
    q_upper = stats.beta.ppf(1 - delta, k_array + 1, N - k_array)

    # Handle the edge case where k == N
    q_upper = np.where(k_array == N, 1.0, q_upper)
    return q_upper


def execute_drs_gatekeeper(
    key,
    state,
    offline_dataset,
    previous_s,
    params,
    nominal_policy_fn,
    active_obstacles,
    pcis_data,
    all_obstacles,
    rollout_plot_count=0,
):
    """
    Evaluates candidates in parallel using JAX, then computes the statistical
    bounds in Python using SciPy to select the optimal switching time.
    """
    M, T, N = params["M"], params["T"], params["N"]
    epsilon, delta, beta = params["epsilon"], params["delta"], params["beta"]
    alpha = pcis_data.get("alpha", 0.0)  # From PCIS backup policy guarantees

    if alpha >= epsilon:
        raise ValueError(
            f"Requirement failed: PCIS probability alpha ({alpha}) must be strictly less than DRCCP threshold epsilon ({epsilon})."
        )

    all_obstacles_np = format_obstacles(all_obstacles)

    pcis_inputs = {
        "states": jnp.asarray(pcis_data["states"]),
        "tolerances": jnp.asarray(pcis_data["tolerances"]),
        "controls": jnp.asarray(pcis_data["controls"]),
        "control_mask": jnp.asarray(pcis_data["control_mask"]),
        "grid": jnp.asarray(pcis_data["grid"]),
        "grid_axes_px": jnp.asarray(pcis_data["grid_axes"][0]),
        "grid_axes_py": jnp.asarray(pcis_data["grid_axes"][1]),
        "grid_axes_theta": jnp.asarray(pcis_data["grid_axes"][2]),
        "grid_axes_v": jnp.asarray(pcis_data["grid_axes"][3]),
        "position_radius": jnp.asarray(pcis_data["position_radius"]),
        "inner_radius": jnp.asarray(pcis_data["inner_radius"]),
        "torus_width": jnp.asarray(pcis_data["torus_width"]),
        "control_grid": jnp.asarray(pcis_data["control_grid"]),
        "control_mask_grid": jnp.asarray(pcis_data["control_mask_grid"]),
        "obstacle_centers": jnp.asarray(all_obstacles_np[:, :2]),
        "obstacle_radii": jnp.asarray(all_obstacles_np[:, 2]),
    }

    # 1. Sample M * N process noise trajectories and M * N obstacle noise vectors
    flat_samples = sample_empirical_disturbances(key, offline_dataset, M * N * T)
    W_process_samples = flat_samples.reshape((M, N, T * DIST_DIM))

    active_obstacles_np = format_obstacles(active_obstacles, allow_empty=True)
    num_obs = active_obstacles_np.shape[0]

    scale = params.get("sensor_noise_scale", 0.0)
    shape_val = params.get("sensor_noise_shape", SENSOR_NOISE_SHAPE)
    if scale > 0 and num_obs > 0:
        obs_key, key = jax.random.split(key)
        obs_noise = generate_skew_normal_noise(obs_key, (M, N, num_obs, 3), scale, shape_val)
        obs_noise = np.clip(obs_noise, -scale * SENSOR_NOISE_CLIP, scale * SENSOR_NOISE_CLIP)
        W_obs_samples = obs_noise.reshape((M, N, num_obs * 3))
    else:
        W_obs_samples = np.zeros((M, N, num_obs * 3))

    W_samples = jnp.concatenate([W_process_samples, W_obs_samples], axis=2)

    if num_obs == 0:
        rollout_samples = None
        if rollout_plot_count > 0:
            rollout_samples = W_samples[-1, :rollout_plot_count]
        q_bar_array = np.full(M, np.nan, dtype=float)
        return int(T), rollout_samples, None, q_bar_array

    obstacle_centers = jnp.asarray(active_obstacles_np[:, :2])
    obstacle_radii = jnp.asarray(active_obstacles_np[:, 2])

    # 2. Define candidate switching times [0, 1, ..., M-1]
    s_array = jnp.arange(M)

    # 3. Evaluate all candidates in JAX
    # Returns a JAX array of shape (M,) containing the failure count k^m for each candidate
    jax_failures, _, _, failure_masks = evaluate_all_candidates(
        state,
        s_array,
        W_samples,
        nominal_policy_fn,
        obstacle_centers,
        obstacle_radii,
        pcis_inputs,
        beta,
    )

    # 4. Transfer counts to CPU for SciPy processing
    k_array = np.array(jax_failures)

    # 5. Compute Clopper-Pearson bounds q_bar
    # sidak correction
    rho = 1 - (1 - delta) ** (1 / M)
    q_bar_array = compute_q_bar(k_array, N, rho)

    # 6. Filter candidates strictly satisfying the DRCC threshold
    adjusted_epsilon = (epsilon - alpha) / (1.0 - alpha)
    valid_mask = q_bar_array <= adjusted_epsilon

    # Extract the indices of valid candidates
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) > 0:
        # Select the maximum valid switching time to maximize nominal policy usage
        best_s_relative = int(np.max(valid_indices))
    else:
        best_s_relative = int(previous_s)

    rollout_samples = None
    rollout_failures = None
    if rollout_plot_count > 0:
        rollout_samples = W_samples[best_s_relative, :rollout_plot_count]
        rollout_failures = np.array(failure_masks[best_s_relative][:rollout_plot_count])

    return best_s_relative, rollout_samples, rollout_failures, q_bar_array


@jax.jit(static_argnames=["nominal_policy_fn"])
def rollout_with_pcis_backup(
    state,
    s,
    W_flat,
    nominal_policy_fn,
    obstacle_centers,
    obstacle_radii,
    pcis_states,
    pcis_tolerances,
    pcis_controls,
    pcis_control_mask,
):
    """Roll out a candidate using the PCIS policy after its switch time."""
    num_obs = obstacle_centers.shape[0]
    obs_noise_dim = num_obs * 3
    if obs_noise_dim > 0:
        W_process_flat = W_flat[:-obs_noise_dim]
        W_obs_flat = W_flat[-obs_noise_dim:]
        W = W_process_flat.reshape((-1, DIST_DIM))
        obs_noise = W_obs_flat.reshape((num_obs, 3))
        noisy_centers = obstacle_centers + obs_noise[:, :2]
        noisy_radii = obstacle_radii + obs_noise[:, 2]
    else:
        W = W_flat.reshape((-1, DIST_DIM))
        noisy_centers = obstacle_centers
        noisy_radii = obstacle_radii

    def scan_step(current_state, inputs):
        t, w_t = inputs
        u = jax.lax.cond(
            t < s,
            lambda _: nominal_policy_fn(current_state),
            lambda _: backup_policy(
                current_state,
                closest_obstacle_center_jax(current_state, noisy_centers, noisy_radii),
            ),
            operand=None,
        )
        next_state = dynamics_step(current_state, u, w_t)
        return next_state, next_state

    time_indices = jnp.arange(W.shape[0])
    _, states = jax.lax.scan(scan_step, state, (time_indices, W))
    return jnp.concatenate([state[jnp.newaxis, :], states], axis=0)


@jax.jit(static_argnames=["nominal_policy_fn"])
def rollout_pcis_bundle(
    state,
    s,
    W_samples,
    nominal_policy_fn,
    obstacle_centers,
    obstacle_radii,
    pcis_states,
    pcis_tolerances,
    pcis_controls,
    pcis_control_mask,
):
    return jax.vmap(
        rollout_with_pcis_backup,
        in_axes=(None, None, 0, None, None, None, None, None, None, None),
    )(
        state,
        s,
        W_samples,
        nominal_policy_fn,
        obstacle_centers,
        obstacle_radii,
        pcis_states,
        pcis_tolerances,
        pcis_controls,
        pcis_control_mask,
    )


def run_gatekeeper_simulation(
    algo_key,
    env_key,
    initial_state,
    dataset,
    params,
    obstacles,
    gatekeeper_visual_radius,
    rollout_plot_count=100,
    verbose=True,
    return_stats=False,
    pcis_data=None,
):
    obstacles_np = format_obstacles(obstacles)
    if pcis_data is None:
        raise ValueError("run_gatekeeper_simulation requires precomputed pcis_data.")

    pcis_states = jnp.asarray(pcis_data["states"])
    pcis_tolerances = jnp.asarray(pcis_data["tolerances"])
    pcis_controls = jnp.asarray(pcis_data["controls"])
    pcis_control_mask = jnp.asarray(pcis_data["control_mask"])
    # The true-process radius is configured independently in helper.TRUE_BETA;
    # It can be overridden via params["true_beta"] during a parameter sweep.
    true_distribution_beta = float(params.get("true_beta", TRUE_BETA))
    true_shift_direction = params.get("true_shift_direction", None)
    true_sequence_metric = params.get("true_sequence_metric", "l2")
    true_sequence_horizon = int(params.get("T", N_STEPS))

    # Draw one coupled process sequence. The transport budget is allocated
    # over the DRS-GK horizon T, so every contiguous T-step window has
    # W_infinity transport cost at most beta under the selected sequence norm.
    _, true_disturbance_sequence = sample_coupled_true_process_sequence(
        env_key,
        dataset,
        n_steps=N_STEPS,
        sequence_horizon=true_sequence_horizon,
        beta=true_distribution_beta,
        shift_direction=true_shift_direction,
        sequence_metric=true_sequence_metric,
    )

    state = initial_state
    state_history = [state]
    control_history = []
    rollout_history = []
    switch_offset_history = []
    rollout_failures_history = []
    time_history = []
    predicted_probs = None
    backup_steps = 0
    total_steps = 0
    hit_obstacle = False
    goal_reached = False
    time_to_goal = np.inf

    nominal_policy_fn = lambda s: nominal_policy(s, target_pos, target_v)

    switching_time = 0  # Initialize switching time to 0 (start with nominal policy)
    is_open_loop = bool(params.get("open_loop", False))
    rollout_samples = None
    rollout_failures = None
    rollout_states = None

    locked_turn_direction = None

    for t in range(N_STEPS):
        # Split the keys independently
        algo_key, gk_key, sensor_key = jax.random.split(algo_key, 3)

        t0 = time.time()

        # Apply sensor noise to obstacles
        scale = params.get("sensor_noise_scale", 0.0)
        shape_val = params.get("sensor_noise_shape", SENSOR_NOISE_SHAPE)
        if scale > 0:
            sensor_noise = generate_skew_normal_noise(sensor_key, obstacles_np.shape, scale, shape_val)
            sensor_noise = np.clip(sensor_noise, -scale * SENSOR_NOISE_CLIP, scale * SENSOR_NOISE_CLIP)
            sensed_obstacles = obstacles_np + sensor_noise
        else:
            sensed_obstacles = obstacles_np

        all_sensed_centers = jnp.asarray(sensed_obstacles[:, :2])
        all_sensed_radii = jnp.asarray(sensed_obstacles[:, 2])

        # 1. Gatekeeper safety check
        active_obstacle_indices = visible_obstacle_indices(state, sensed_obstacles, gatekeeper_visual_radius)
        active_obstacles = sensed_obstacles[active_obstacle_indices]
        if t == 0 or not is_open_loop:
            switch_offset, rollout_samples, rollout_failures, q_bar_array = execute_drs_gatekeeper(
                gk_key,
                state,
                dataset,
                switching_time - t,
                params,
                nominal_policy_fn,
                active_obstacles,
                pcis_data,
                sensed_obstacles,
                rollout_plot_count=rollout_plot_count,
            )
            predicted_probs = q_bar_array
            switching_time = t + switch_offset
            if rollout_samples is not None:
                rollout_states = rollout_pcis_bundle(
                    state,
                    switch_offset,
                    rollout_samples,
                    nominal_policy_fn,
                    all_sensed_centers,
                    all_sensed_radii,
                    pcis_states,
                    pcis_tolerances,
                    pcis_controls,
                    pcis_control_mask,
                )
        else:
            switch_offset = switching_time

        t1 = time.time() - t0
        time_history.append(t1)
        if verbose:
            print(
                f"Step {t}: Selected switching time = {switching_time}, Gatekeeper computation time = {t1:.3f} seconds"
            )

        if rollout_states is not None:
            rollout_history.append(np.array(rollout_states))
            switch_offset_history.append(int(switch_offset))
            rollout_failures_history.append(rollout_failures)

        # 2. Policy selection
        if t < switching_time:
            u_applied = nominal_policy_fn(state)
        else:
            if is_open_loop:
                if locked_turn_direction is None:
                    closest = closest_obstacle_center_jax(state, all_sensed_centers, all_sensed_radii)
                    rel_px = state[0] - closest[0]
                    rel_py = state[1] - closest[1]
                    turn_metric = rel_py * np.cos(state[2]) - rel_px * np.sin(state[2])
                    locked_turn_direction = np.sign(turn_metric + 1e-6)
                u_applied = backup_policy(
                    state,
                    closest_obstacle_center_jax(state, all_sensed_centers, all_sensed_radii),
                    locked_turn_direction,
                )
            else:
                u_applied = backup_policy(
                    state,
                    closest_obstacle_center_jax(state, all_sensed_centers, all_sensed_radii),
                )
            backup_steps += 1

        # 3. True disturbance realization
        # Draw from the shifted true distribution to represent reality.
        true_disturbance = true_disturbance_sequence[t]

        # 4. Step dynamics
        state = dynamics_step(state, u_applied, true_disturbance)

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
        rollout_failures_history,
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
    if predicted_probs is not None:
        stats["predicted_probs"] = np.array(predicted_probs).tolist()
    return (*result, stats)


def get_drs_gk_bounds(
    key,
    state,
    dataset,
    params,
    obstacles_np,
    gatekeeper_visual_radius,
    pcis_data=None,
):
    """Executes a single DRS-Gatekeeper sampling iteration to compute failure probability bounds."""
    gk_key, sensor_key = jax.random.split(key)

    scale = params.get("sensor_noise_scale", 0.0)
    shape_val = params.get("sensor_noise_shape", SENSOR_NOISE_SHAPE)
    if scale > 0:
        sensor_noise = generate_skew_normal_noise(sensor_key, obstacles_np.shape, scale, shape_val)
        sensor_noise = np.clip(sensor_noise, -scale * SENSOR_NOISE_CLIP, scale * SENSOR_NOISE_CLIP)
        sensed_obstacles = obstacles_np + sensor_noise
    else:
        sensed_obstacles = obstacles_np

    active_obstacle_indices = visible_obstacle_indices(state, sensed_obstacles, gatekeeper_visual_radius)
    active_obstacles = sensed_obstacles[active_obstacle_indices]
    if active_obstacles.shape[0] == 0:
        closest_idx = closest_obstacle_index(state, sensed_obstacles)
        active_obstacles = sensed_obstacles[[closest_idx]]

    M = params.get("M", 50)
    nominal_policy_fn = lambda s: nominal_policy(s, target_pos, target_v)
    switch_offset, _, _, q_bar_array = execute_drs_gatekeeper(
        gk_key,
        state,
        dataset,
        M,
        params,
        nominal_policy_fn,
        active_obstacles,
        pcis_data,
        sensed_obstacles,
        rollout_plot_count=0,
    )
    return switch_offset, q_bar_array
