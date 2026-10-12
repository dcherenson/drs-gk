import numpy as np
from math import ceil, log
import jax.numpy as jnp
import jax
import time
from config import (
    CONTROL_DIM,
    DEFAULT_GOAL_TOLERANCE,
    DIST_DIM,
    DT,
    N_STEPS,
    SENSOR_NOISE_CLIP,
    SENSOR_NOISE_SHAPE,
    TRUE_BETA,
    target_pos,
    target_v,
)
from helper import (
    backup_policy,
    closest_obstacle_center_jax,
    collides_with_any_obstacle,
    dynamics_step,
    format_obstacles,
    generate_skew_normal_noise,
    nominal_policy,
    rollout_bundle,
    sample_coupled_true_process_sequence,
    sample_empirical_disturbances,
    visible_obstacle_indices,
)
from controller.drs_gatekeeper import (
    pcis_grid_terminal_margin,
)


def compute_smps_K(t, epsilon, delta):
    """
    Computes K_t according to the SMPS statistical guarantee.
    """

    epsilon_t = 6.0 * epsilon / ((t + 1) ** 2 * np.pi**2)
    delta_t = 6.0 * delta / ((t + 1) ** 2 * np.pi**2)

    K = int(ceil(log(1.0 / delta_t) / log(1.0 / (1.0 - epsilon_t))))

    return K, epsilon_t, delta_t


@jax.jit(static_argnames=["nominal_policy_fn"])
def compute_H(
    state,
    W,
    nominal_policy_fn,
    obstacle_centers,
    obstacle_radii,
    pcis_inputs,
):
    def scan_step(carry, inputs):
        current_state = carry
        t, w_t = inputs

        # Select policy based on switching time s
        u = jax.lax.cond(
            t < 1,
            lambda _: nominal_policy_fn(current_state),
            lambda _: backup_policy(
                current_state,
                closest_obstacle_center_jax(current_state, obstacle_centers, obstacle_radii),
            ),
            operand=None,
        )

        next_state = dynamics_step(current_state, u, w_t)

        distances = jnp.linalg.norm(obstacle_centers - next_state[:2], axis=1)
        margin = jnp.min(distances - obstacle_radii)

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

    # 2. Terminal PCIS margin (h_c)
    H_terminal = pcis_grid_terminal_margin(
        final_state,
        obstacle_centers,
        obstacle_radii,
        pcis_inputs,
    )

    # The final safety function is the minimum of the trajectory margin and terminal margin
    H_T = jnp.minimum(H_traj, H_terminal)
    return H_T


@jax.jit(static_argnames=["nominal_policy_fn"])
def is_rec_single(
    state,
    W,
    nominal_policy_fn,
    obstacle_centers,
    obstacle_radii,
    pcis_inputs,
):
    H_T = compute_H(
        state,
        W,
        nominal_policy_fn,
        obstacle_centers,
        obstacle_radii,
        pcis_inputs,
    )
    return H_T >= 0.0


@jax.jit(static_argnames=["nominal_policy_fn"])
def is_rec(
    state,
    W_samples,
    nominal_policy_fn,
    obstacle_centers,
    obstacle_radii,
    pcis_inputs,
):
    recoverability = jax.vmap(is_rec_single, in_axes=(None, 0, None, None, None, None))(
        state,
        W_samples,
        nominal_policy_fn,
        obstacle_centers,
        obstacle_radii,
        pcis_inputs,
    )
    return jnp.all(recoverability)


# ============================================================
# Execute SMPS
# ============================================================


def execute_smps(
    key,
    state,
    offline_dataset,
    t,
    params,
    nominal_policy_fn,
    active_obstacles,
    pcis_data,
    all_obstacles,
    rollout_plot_count=0,
):
    """
    Executes SMPS at the CURRENT state.

    The output is the control action for the current time step only.

    If the sampled state is recoverable:
        use nominal policy.

    Otherwise:
        use the same backup policy as the DRS-gatekeeper.

    SMPS does NOT commit to a future switching time.
    """

    T = params["T"]
    epsilon = params["epsilon"]
    delta = params["delta"]

    # --------------------------------------------------------
    # Compute number of disturbance trajectories
    # --------------------------------------------------------

    K, _, _ = compute_smps_K(0, epsilon, delta)

    # --------------------------------------------------------
    # Prepare active obstacles
    # --------------------------------------------------------
    active_obstacles_np = format_obstacles(active_obstacles, allow_empty=True)

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

    # No obstacles in the active set
    if active_obstacles_np.shape[0] == 0:
        u = nominal_policy_fn(state)
        return u, True, K, None

    obstacle_centers = jnp.asarray(active_obstacles_np[:, :2])
    obstacle_radii = jnp.asarray(active_obstacles_np[:, 2])

    # --------------------------------------------------------
    # Sample K disturbance trajectories
    #
    # Each trajectory has N+1 disturbance samples.
    # --------------------------------------------------------
    # These are the nominal predictive samples used by SMPS. They are also
    # returned for visualization; the true closed-loop process is sampled
    # separately in run_smps_simulation.
    key, sample_key = jax.random.split(key)

    flat_samples = sample_empirical_disturbances(sample_key, offline_dataset, K * (T + 1))

    W_samples = flat_samples.reshape((K, T + 1, DIST_DIM))
    rollout_samples = W_samples[:rollout_plot_count] if rollout_plot_count > 0 else None

    # --------------------------------------------------------
    # Statistical recoverability test
    # --------------------------------------------------------

    recoverable = is_rec(
        state,
        W_samples,
        nominal_policy_fn,
        obstacle_centers,
        obstacle_radii,
        pcis_inputs,
    )

    recoverable = bool(recoverable)

    # --------------------------------------------------------
    # Select action for CURRENT time step only
    # --------------------------------------------------------

    if recoverable:
        u = nominal_policy_fn(state)
    else:
        u = backup_policy(
            state,
            closest_obstacle_center_jax(state, obstacle_centers, obstacle_radii),
        )

    return u, recoverable, K, rollout_samples


# ============================================================
# Run SMPS closed-loop simulation
# ============================================================


def run_smps_simulation(
    algo_key,
    env_key,
    initial_state,
    dataset,
    params,
    obstacles,
    gatekeeper_visual_radius,
    pcis_data=None,
    verbose=True,
    return_stats=False,
    rollout_plot_count=100,
):
    """
    Runs closed-loop simulation using SMPS.

    At every time step:

        1. Find active obstacles.
        2. Run SMPS.
        3. SMPS selects nominal or backup for THIS step.
        4. Apply the selected action.
        5. Sample the true disturbance.
        6. Propagate the dynamics.
        7. Repeat.

    Unlike DRS-gatekeeper, there is no persistent switching_time.
    """

    obstacles_np = format_obstacles(obstacles)

    state = initial_state

    state_history = [state]
    control_history = []
    rollout_history = []
    time_history = []

    backup_steps = 0
    total_steps = 0

    hit_obstacle = False
    goal_reached = False

    time_to_goal = np.inf

    nominal_policy_fn = lambda s: nominal_policy(s, target_pos, target_v)

    true_sequence_horizon = int(params.get("T", N_STEPS))
    true_shift_direction = params.get("true_shift_direction", None)
    true_sequence_metric = params.get("true_sequence_metric", "l2")

    # SMPS predictive recoverability samples remain nominal inside
    # execute_smps. This sequence is used only by the realized closed-loop
    # dynamics below.
    _, true_disturbance_sequence = sample_coupled_true_process_sequence(
        env_key,
        dataset,
        n_steps=N_STEPS,
        sequence_horizon=true_sequence_horizon,
        beta=TRUE_BETA,
        shift_direction=true_shift_direction,
        sequence_metric=true_sequence_metric,
    )

    recoverability_history = []
    K_history = []

    for t in range(N_STEPS):
        algo_key, smps_key = jax.random.split(algo_key)
        t0 = time.time()

        # Apply sensor noise to obstacles
        scale = params.get("sensor_noise_scale", 0.0)
        shape_val = params.get("sensor_noise_shape", SENSOR_NOISE_SHAPE)
        if scale > 0:
            algo_key, sensor_key = jax.random.split(algo_key)
            sensor_noise = generate_skew_normal_noise(sensor_key, obstacles_np.shape, scale, shape_val)
            sensor_noise = np.clip(
                sensor_noise,
                -scale * SENSOR_NOISE_CLIP,
                scale * SENSOR_NOISE_CLIP,
            )
            sensed_obstacles = obstacles_np + sensor_noise
        else:
            sensed_obstacles = obstacles_np

        # ----------------------------------------------------
        # Determine active obstacles
        # ----------------------------------------------------

        active_obstacle_indices = visible_obstacle_indices(state, sensed_obstacles, gatekeeper_visual_radius)

        active_obstacles = sensed_obstacles[active_obstacle_indices]

        # ----------------------------------------------------
        # SMPS
        # ----------------------------------------------------

        u_applied, recoverable, K, rollout_samples = execute_smps(
            smps_key,
            state,
            dataset,
            t,
            params,
            nominal_policy_fn,
            active_obstacles,
            pcis_data,
            obstacles,
            rollout_plot_count=rollout_plot_count,
        )

        if rollout_samples is not None:
            rollout_states = rollout_bundle(
                state,
                1,
                rollout_samples,
                nominal_policy_fn,
                jnp.asarray(sensed_obstacles[:, :2]),
                jnp.asarray(sensed_obstacles[:, 2]),
            )
            rollout_history.append(np.array(rollout_states))

        computation_time = time.time() - t0
        time_history.append(computation_time)
        recoverability_history.append(recoverable)
        K_history.append(K)

        if recoverable:
            policy_name = "nominal"
        else:
            policy_name = "backup"
            backup_steps += 1

        if verbose:
            print(f"Step {t}: SMPS = {policy_name}, K = {K}, computation time = {computation_time:.3f} seconds")

        # ----------------------------------------------------
        # Use the shifted true disturbance only for the realized dynamics.
        # ----------------------------------------------------

        true_disturbance = true_disturbance_sequence[t]

        # ----------------------------------------------------
        # Apply selected control
        # ----------------------------------------------------

        state = dynamics_step(state, u_applied, true_disturbance)

        state_history.append(state)
        control_history.append(u_applied)

        total_steps += 1

        # ----------------------------------------------------
        # Collision check
        # ----------------------------------------------------

        if collides_with_any_obstacle(state, obstacles_np):
            hit_obstacle = True

            if verbose:
                print(f"Collision detected at step {t} with state {state[:2]}")

            break

        # ----------------------------------------------------
        # Goal check
        # ----------------------------------------------------

        if jnp.linalg.norm(state[:2] - target_pos) <= DEFAULT_GOAL_TOLERANCE:
            goal_reached = True
            time_to_goal = float((t + 1) * DT)

            if verbose:
                print(f"Goal reached at step {t} with state {state[:2]}")

            break

    # ========================================================
    # Convert histories to arrays
    # ========================================================

    controls_array = (
        jnp.stack(control_history) if control_history else jnp.empty((0, CONTROL_DIM), dtype=state_history[0].dtype)
    )

    result = (jnp.stack(state_history), controls_array, rollout_history, recoverability_history)

    if not return_stats:
        return result

    backup_usage_pct = 100.0 * backup_steps / total_steps if total_steps > 0 else 0.0

    stats = {
        "backup_usage_pct": float(backup_usage_pct),
        "hit_obstacle": bool(hit_obstacle),
        "time_to_goal": float(time_to_goal) if goal_reached else float(np.inf),
        "total_steps": int(total_steps),
        "backup_steps": int(backup_steps),
        "mean_K": float(np.mean(K_history)) if K_history else 0.0,
        "mean_computation_time": float(np.mean(time_history)) if time_history else 0.0,
    }

    return (*result, stats)
