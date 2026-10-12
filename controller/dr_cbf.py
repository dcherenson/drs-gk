import time
import jax
import jax.numpy as jnp
import numpy as np
import cvxpy as cp

from config import (
    A_MAX,
    A_MIN,
    CONTROL_DIM,
    DT,
    GOAL_TOLERANCE,
    N_STEPS,
    SENSOR_NOISE_CLIP,
    SENSOR_NOISE_SHAPE,
    TRUE_BETA,
    W_MAX,
    cbf_params,
    gk_params,
    target_pos,
    target_v,
)
from helper import (
    collides_with_any_obstacle,
    closest_obstacle_index,
    dynamics_step,
    format_obstacles,
    generate_skew_normal_noise,
    nominal_policy,
    sample_coupled_true_process_sequence,
)


def solve_drccp_cbf(
    state,
    u_nom,
    dataset,
    obstacle_center,
    obstacle_radius,
    alpha1=cbf_params["alpha1"],
    epsilon=cbf_params["epsilon"],
    radius=cbf_params["beta"],
):
    state = np.array(state)
    u_nom = np.array(u_nom)
    dataset = np.array(dataset)
    obstacle_center = np.array(obstacle_center)

    px, py, theta, v = state[0], state[1], state[2], state[3]
    rel_px, rel_py = px - obstacle_center[0], py - obstacle_center[1]
    N = dataset.shape[0]

    xi_a = dataset[:, 0]
    xi_w = dataset[:, 1]
    eta_a = dataset[:, 2]
    eta_w = dataset[:, 3]

    alpha2_var = cp.Variable(1, nonneg=True)

    A = 2.0 * (rel_px * np.cos(theta) + rel_py * np.sin(theta))
    B = 2.0 * v * (rel_py * np.cos(theta) - rel_px * np.sin(theta))
    C = 2.0 * v**2 + alpha1 * (A * v)

    h_val = rel_px**2 + rel_py**2 - obstacle_radius**2
    h_e = (A * v) + alpha1 * h_val

    u = cp.Variable(2)
    y = cp.Variable(1)
    t = cp.Variable(1)
    s = cp.Variable(N)
    L_vec = cp.hstack([A * v, B * v, A * u[0], B * u[1]])
    L = cp.norm(L_vec, 2)

    CBC = A * u[0] + B * u[1] + A * v * xi_a + B * v * xi_w + A * u[0] * eta_a + B * u[1] * eta_w + C + alpha2_var * h_e
    G = -CBC

    objective = cp.Minimize(y + cp.abs(alpha2_var))

    soc_term = cp.hstack([2 * (u - u_nom), y - 1])

    constraints = [
        radius * L + (1.0 / N) * cp.sum(s) - t * epsilon <= 0,
        s >= G + t,
        s >= 0,
        y + 1 >= cp.norm(soc_term, 2),
        u[0] >= A_MIN,
        u[0] <= A_MAX,
        u[1] >= -W_MAX,
        u[1] <= W_MAX,
    ]
    prob = cp.Problem(objective, constraints)

    try:
        prob.solve()
    except:
        return False, np.array(u_nom)

    if prob.status in ["optimal", "optimal_inaccurate"]:
        return True, np.array(u.value)
    else:
        return False, np.array(u_nom)


def run_cbf_simulation(env_key, initial_state, dataset, obstacles, params=None, verbose=True, return_stats=False):
    obstacles_np = format_obstacles(obstacles)
    params = params or {}
    alpha1 = float(params.get("alpha1", cbf_params["alpha1"]))
    epsilon = float(params.get("epsilon", cbf_params["epsilon"]))
    beta = float(params.get("beta", cbf_params["beta"]))
    sample_count = int(params.get("sample_count", cbf_params["sample_count"]))
    true_sequence_horizon = int(params.get("T", gk_params.get("T", N_STEPS)))
    true_shift_direction = params.get("true_shift_direction", None)
    true_sequence_metric = params.get("true_sequence_metric", "l2")

    # This is the closed-loop process sequence. The nominal dataset remains
    # unchanged for the CBF optimization; only the realized dynamics use the
    # shifted true distribution.
    _, true_disturbance_sequence = sample_coupled_true_process_sequence(
        env_key,
        dataset,
        n_steps=N_STEPS,
        sequence_horizon=true_sequence_horizon,
        beta=TRUE_BETA,
        shift_direction=true_shift_direction,
        sequence_metric=true_sequence_metric,
    )

    state = initial_state
    state_history = [state]
    control_history = []
    hit_obstacle = False
    infeasible = False
    goal_reached = False
    time_to_goal = np.inf
    time_history = []
    for t in range(N_STEPS):
        # 0. Apply sensor noise
        env_key, sensor_key = jax.random.split(env_key)
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

        # 1. Compute nominal control
        t0 = time.time()
        u_nom = nominal_policy(state, target_pos, target_v)

        # 2. Apply CBF safety filter
        closest_idx = closest_obstacle_index(state, sensed_obstacles)
        closest_obstacle = sensed_obstacles[closest_idx]
        is_feasible, u_safe = solve_drccp_cbf(
            state,
            u_nom,
            dataset[:sample_count, :],
            closest_obstacle[:2],
            float(closest_obstacle[2]),
            alpha1=alpha1,
            epsilon=epsilon,
            radius=beta,
        )
        if not is_feasible:
            infeasible = True
            if verbose:
                print(f"CBF infeasible at step {t}")
            break

        # 3. True disturbance realization. This is not used by the predictive
        # CBF optimization; it only affects the simulated system.
        true_disturbance = true_disturbance_sequence[t]
        time_history.append(time.time() - t0)

        # 4. Step dynamics
        state = dynamics_step(state, u_safe, true_disturbance)

        # Log data
        state_history.append(state)
        control_history.append(u_safe)

        # check collision
        if collides_with_any_obstacle(state, obstacles_np):
            hit_obstacle = True
            if verbose:
                print(f"Collision detected at step {t} with state {state[:2]}")
            break

        # check if goal is reached
        if jnp.linalg.norm(state[:2] - target_pos) <= GOAL_TOLERANCE:
            goal_reached = True
            time_to_goal = float((t + 1) * DT)
            if verbose:
                print(f"Goal reached at step {t} with state {state[:2]}")
            break

    controls_array = (
        jnp.stack(control_history) if control_history else jnp.empty((0, CONTROL_DIM), dtype=state_history[0].dtype)
    )
    result = (jnp.stack(state_history), controls_array)
    if not return_stats:
        return result

    stats = {
        "hit_obstacle": bool(hit_obstacle),
        "infeasible": bool(infeasible),
        "time_to_goal": float(np.inf) if infeasible else (float(time_to_goal) if goal_reached else float(np.inf)),
        "mean_computation_time": float(np.mean(time_history)) if time_history else 0.0,
    }
    return (*result, stats)
