from __future__ import annotations

import time

import cvxpy as cp
import numpy as np

import jax
import jax.numpy as jnp

from config import (
    A_MAX,
    A_MIN,
    DIST_DIM,
    K_P,
    K_THETA,
    SENSOR_NOISE_CLIP,
    SENSOR_NOISE_SHAPE,
    TRUE_BETA,
    WT_MPC_GOAL_TOLERANCE,
    WT_MPC_SLOWDOWN_DISTANCE,
    WT_MPC_OBSTACLE_MARGIN,
    W_MAX,
    W_MIN,
    PolyhedralConstraints,  # noqa: F401 - preserve the WT-MPC type export from config.
    WTMPCConfig,
    target_pos,
    target_v,
)
from helper import (
    dynamics_step,
    generate_skew_normal_noise,
    sample_coupled_true_process_sequence,
    sample_empirical_disturbances,
)


# ============================================================
# DUBINS LINEARIZATION
#
# x_{k+1} = f(x_k, u_k)
#
# Linearization:
#
# x_{k+1}
#   ~= A_k x_k + B_k u_k + c_k
#
# where
#
# A_k = df/dx
# B_k = df/du
#
# ============================================================


def linearize_dubins(state: np.ndarray, control: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    px, py, theta, v = state
    a, omega = control

    A = np.array(
        [
            [
                1.0,
                0.0,
                -dt * v * np.sin(theta),
                dt * np.cos(theta),
            ],
            [
                0.0,
                1.0,
                dt * v * np.cos(theta),
                dt * np.sin(theta),
            ],
            [
                0.0,
                0.0,
                1.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                1.0,
            ],
        ]
    )

    B = np.array(
        [
            [
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
            ],
            [
                0.0,
                dt,
            ],
            [
                dt,
                0.0,
            ],
        ]
    )

    f_nominal = np.asarray(dynamics_step(jnp.asarray(state), jnp.asarray(control), jnp.zeros(4)))

    c = f_nominal - A @ state - B @ control

    return A, B, c


# ============================================================
# NOMINAL TRAJECTORY LINEARIZATION
# ============================================================


def linearize_nominal_trajectory(
    nominal_states: np.ndarray, nominal_controls: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    horizon = nominal_controls.shape[0]

    A_seq = np.zeros((horizon, 4, 4))
    B_seq = np.zeros((horizon, 4, 2))
    c_seq = np.zeros((horizon, 4))

    for k in range(horizon):
        A_seq[k], B_seq[k], c_seq[k] = linearize_dubins(nominal_states[k], nominal_controls[k], dt)

    return A_seq, B_seq, c_seq


# ============================================================
# TIME-VARYING LQR
#
# We use:
#
#     u_k = v_k + K_k e_k
#
# where:
#
#     e_k = x_k - z_k
#
# ============================================================


def compute_time_varying_lqr_gains(
    A_seq: np.ndarray, B_seq: np.ndarray, Q: np.ndarray, R: np.ndarray, Qf: np.ndarray
) -> np.ndarray:

    horizon = A_seq.shape[0]
    nx = A_seq.shape[1]
    nu = B_seq.shape[2]

    P = Qf.copy()

    K_seq = np.zeros((horizon, nu, nx))

    for k in range(horizon - 1, -1, -1):
        A_k = A_seq[k]
        B_k = B_seq[k]

        S = R + B_k.T @ P @ B_k

        K_k = -np.linalg.solve(S, B_k.T @ P @ A_k)

        K_seq[k] = K_k

        P = Q + A_k.T @ P @ A_k + A_k.T @ P @ B_k @ K_k

    return K_seq


# ============================================================
# CLOSED LOOP ERROR DYNAMICS
#
# e_{k+1}
#   = A_k e_k + B_k K_k e_k + w_k
#
#   = Acl_k e_k + w_k
#
# ============================================================


def compute_closed_loop_matrices(A_seq: np.ndarray, B_seq: np.ndarray, K_seq: np.ndarray) -> np.ndarray:

    horizon = A_seq.shape[0]

    Acl_seq = np.zeros_like(A_seq)

    for k in range(horizon):
        Acl_seq[k] = A_seq[k] + B_seq[k] @ K_seq[k]

    return Acl_seq


# ============================================================
# ERROR SAMPLE PROPAGATION
#
# e_0 = 0
#
# e_{k+1}
#   = Acl_k e_k + w_k
#
# ============================================================


def propagate_error_samples(W_samples: np.ndarray, Acl_seq: np.ndarray) -> np.ndarray:

    W_samples = np.asarray(W_samples, dtype=float)

    n_samples, horizon, nx = W_samples.shape

    if Acl_seq.shape != (horizon, nx, nx):
        raise ValueError(f"Acl_seq must have shape {(horizon, nx, nx)}, got {Acl_seq.shape}.")

    error_samples = np.zeros((n_samples, horizon + 1, nx))

    for i in range(n_samples):
        for k in range(horizon):
            error_samples[i, k + 1] = Acl_seq[k] @ error_samples[i, k] + W_samples[i, k]

    return error_samples


# ============================================================
# LTV TRANSITION MATRICES
#
# Phi[k, p] =
#
#     Acl[k-1] ... Acl[p]
#
# for p < k
#
# Phi[k, k] = I
#
# ============================================================


def build_ltv_transition_products(Acl_seq: np.ndarray) -> np.ndarray:

    horizon = Acl_seq.shape[0]
    nx = Acl_seq.shape[1]

    Phi = np.zeros((horizon + 1, horizon + 1, nx, nx))

    for k in range(horizon + 1):
        Phi[k, k] = np.eye(nx)

        for p in range(k - 1, -1, -1):
            Phi[k, p] = Phi[k, p + 1] @ Acl_seq[p]

    return Phi


# ============================================================
# D_k MATRIX
#
# The error is:
#
# e_k = D_k w_{0:k-1}
#
# where
#
# D_k =
#
# [ Acl_{k-1} ... Acl_1 Acl_0
#   Acl_{k-1} ... Acl_1       0
#   ...
#   I                         0 ]
#
# More explicitly:
#
# e_k =
#   Phi[k,1] w_0
#   + Phi[k,2] w_1
#   + ...
#   + Phi[k,k] w_{k-1}
#
# ============================================================


def build_D_k(Acl_seq: np.ndarray, k: int) -> np.ndarray:

    nx = Acl_seq.shape[1]

    if k == 0:
        return np.zeros((nx, 0))

    D_k = np.zeros((nx, nx * k))

    for r in range(k):
        propagation = np.eye(nx)

        for j in range(r + 1, k):
            propagation = Acl_seq[j] @ propagation

        D_k[:, r * nx : (r + 1) * nx] = propagation

    return D_k


# ============================================================
# PROPOSITION 2 MATRIX
#
# This is ONLY the matrix appearing in the final norm
# constraint of Proposition 2.
#
# Given D:
#
#     M_D =
#       D^dagger
#       ((D^dagger)^T D^dagger)^(-1)
#
# We use a pseudoinverse for numerical robustness.
#
# IMPORTANT:
#
# This is NOT the entire set of Proposition 2 constraints.
#
# ============================================================


def proposition2_matrix(D: np.ndarray, regularization: float = 1e-9) -> np.ndarray:

    if D.shape[1] == 0:
        return np.zeros((0, D.shape[0]))

    D_dagger = np.linalg.pinv(D)

    G = D_dagger.T @ D_dagger

    G_reg = G + regularization * np.eye(G.shape[0])

    return D_dagger @ np.linalg.inv(G_reg)


# ============================================================
# BOX SUPPORT
#
# W = {w : -w_max <= w <= w_max}
#
# support_W(d)
#   = max_{w in W} d^T w
#   = sum_i w_max_i |d_i|
#
# ============================================================


def box_support(direction: np.ndarray, w_max: np.ndarray) -> float:

    direction = np.asarray(direction, dtype=float).reshape(-1)
    w_max = np.asarray(w_max, dtype=float).reshape(-1)

    return float(np.sum(w_max * np.abs(direction)))


# ============================================================
# BOX SUPPORT MATRICES
#
# Fw <= g
#
# F = [ I ]
#     [-I]
#
# g = [ w_max ]
#     [ w_max ]
#
# ============================================================


def build_box_support(w_max: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    w_max = np.asarray(w_max, dtype=float).reshape(-1)

    nx = len(w_max)

    F = np.vstack([np.eye(nx), -np.eye(nx)])

    g = np.concatenate([w_max, w_max])

    return F, g


# ============================================================
# PROPAGATED DISTURBANCE SUPPORT
# ============================================================


def propagated_disturbance_support(alpha: np.ndarray, p: int, k: int, Phi: np.ndarray, w_max: np.ndarray) -> float:

    if k <= p:
        return 0.0

    value = 0.0

    for r in range(p, k):
        propagation = Phi[k, r + 1]
        direction = propagation.T @ alpha
        value += box_support(direction, w_max)

    return value


# ============================================================
# RECURSIVE STATE TIGHTENING
#
# This is used for the recursive tube constraint.
#
# The tightening at time k is:
#
# max_p support of propagated disturbance.
#
# ============================================================


def compute_recursive_tightening(
    A_constraints: np.ndarray, gamma: float, Phi: np.ndarray, w_max: np.ndarray, horizon: int
) -> np.ndarray:

    n_constraints = A_constraints.shape[0]

    tightening = np.zeros((horizon + 1, n_constraints))

    for k in range(1, horizon + 1):
        for j in range(n_constraints):
            alpha_j = A_constraints[j] / gamma

            max_support = 0.0

            for p in range(1, k + 1):
                support_value = propagated_disturbance_support(alpha_j, p, k, Phi, w_max)

                max_support = max(max_support, support_value)

            tightening[k, j] = max_support

    return tightening


# ============================================================
# INPUT TIGHTENING
#
# u_k = v_k + K_k e_k
#
# H_u u_k <= h_u
#
# ============================================================


def compute_input_tightening(
    Hu: np.ndarray, K_seq: np.ndarray, Phi: np.ndarray, w_max: np.ndarray, horizon: int
) -> np.ndarray:

    n_constraints = Hu.shape[0]

    tightening = np.zeros((horizon, n_constraints))

    for k in range(horizon):
        for j in range(n_constraints):
            direction = K_seq[k].T @ Hu[j]

            value = 0.0

            for r in range(k):
                propagation = Phi[k, r + 1]

                value += box_support(propagation.T @ direction, w_max)

            tightening[k, j] = value

    return tightening


# ============================================================
# DR-CVaR CONSTRAINTS
#
# We impose:
#
# sup_Q in S_t CVaR_{1-gamma}
#     ( max_j a_j^T x_t + b_j )
# <= 0
#
# with:
#
# x_t = z_t + e_t
#
# Proposition 2 gives:
#
# alpha_j^T(z_t + e_i)
# + beta_j(tau)
# + zeta_ij^T(g - F e_i)
# <= s_i
#
# and
#
# || M_D (F^T zeta_ij - alpha_j) ||_2
# <= lambda
#
# together with
#
# lambda * epsilon * N + sum_i s_i <= 0
#
# ============================================================


def add_dr_cvar_constraints(
    constraints,
    z_k,
    error_samples_k,
    A_constraints,
    b_constraints,
    F,
    g,
    D_k,
    gamma,
    epsilon,
    tau,
    lam,
    s,
    zeta_k,
    tightening,
):

    n_samples = error_samples_k.shape[0]

    n_state_constraints = A_constraints.shape[0]

    nx = A_constraints.shape[1]

    if error_samples_k.shape[1] != nx:
        raise ValueError(f"error_samples_k has dimension {error_samples_k.shape[1]}, expected {nx}.")

    if F.shape[1] != nx:
        raise ValueError(f"F must have {nx} columns, got {F.shape[1]}.")

    if g.shape[0] != F.shape[0]:
        raise ValueError(f"g must have {F.shape[0]} entries, got {g.shape[0]}.")

    if zeta_k.shape != (n_samples, n_state_constraints, F.shape[0]):
        raise ValueError(f"zeta_k must have shape {(n_samples, n_state_constraints, F.shape[0])}, got {zeta_k.shape}.")

    M_k = proposition2_matrix(D_k)

    for i in range(n_samples):
        e_i = np.asarray(error_samples_k[i], dtype=float)
        constraints.append(s[i] >= tau)
        for j in range(n_state_constraints):
            alpha_j = A_constraints[j] / gamma

            beta_j = (b_constraints[j] + gamma * tau - tau) / gamma

            zeta_ij = zeta_k[i, j, :]

            constraints.append(alpha_j @ (z_k + e_i) + beta_j + zeta_ij @ (g - F @ e_i) <= s[i] - tightening[j])

            if M_k.shape[0] > 0:
                constraints.append(cp.norm(M_k @ (F.T @ zeta_ij - alpha_j), 2) <= lam)

    constraints.append(lam * epsilon * n_samples + cp.sum(s) <= 0)


def get_linearized_obstacle_constraints(
    nominal_state: np.ndarray,
    obstacles: np.ndarray,
    safety_margin: float = WT_MPC_OBSTACLE_MARGIN,
) -> tuple[np.ndarray, np.ndarray]:

    if obstacles.shape[0] == 0:
        return np.zeros((0, 4)), np.zeros(0)

    px, py = nominal_state[0], nominal_state[1]
    n_obs = obstacles.shape[0]

    A_obs = np.zeros((n_obs, 4))
    b_obs = np.zeros(n_obs)

    for i in range(n_obs):
        cx, cy, r = obstacles[i]

        dx = px - cx
        dy = py - cy
        dist = np.hypot(dx, dy)

        if dist < 1e-3:
            nx, ny = 1.0, 0.0
        else:
            nx, ny = dx / dist, dy / dist

        A_obs[i, 0] = -nx
        A_obs[i, 1] = -ny
        # Inflate the obstacle radius by the safety margin
        b_obs[i] = nx * cx + ny * cy + (r + safety_margin)

    return A_obs, b_obs


# ============================================================
# INITIAL NOMINAL TRAJECTORY
#
# Goal is only:
#
#     goal = [goal_x, goal_y]
#
# We use:
#
#     desired heading
#     proportional acceleration
#     proportional heading-rate
#
# and target velocity.
# ============================================================


def initialize_nominal_trajectory(state: np.ndarray, goal: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:

    state = np.asarray(state, dtype=float)
    goal = np.asarray(goal, dtype=float)

    if state.shape != (4,):
        raise ValueError(f"Expected state shape (4,), got {state.shape}.")

    if goal.shape[0] < 2:
        raise ValueError("Goal must contain [x, y].")

    nominal_states = np.zeros((horizon + 1, 4))
    nominal_controls = np.zeros((horizon, 2))

    nominal_states[0] = state

    for k in range(horizon):
        position_error = goal[:2] - nominal_states[k, :2]

        desired_heading = np.arctan2(position_error[1], position_error[0])

        heading_error = np.arctan2(
            np.sin(desired_heading - nominal_states[k, 2]), np.cos(desired_heading - nominal_states[k, 2])
        )

        distance_error = np.linalg.norm(position_error)

        acceleration = np.clip(K_P * (target_v - nominal_states[k, 3]), A_MIN, A_MAX)

        if distance_error < WT_MPC_SLOWDOWN_DISTANCE:
            acceleration = np.clip(K_P * (distance_error - nominal_states[k, 3]), A_MIN, A_MAX)

        steering_rate = np.clip(K_THETA * heading_error, W_MIN, W_MAX)

        nominal_controls[k] = np.array([acceleration, steering_rate])

        nominal_states[k + 1] = np.asarray(
            dynamics_step(jnp.asarray(nominal_states[k]), jnp.asarray(nominal_controls[k]), jnp.zeros(4))
        )

    return nominal_states, nominal_controls


# ============================================================
# BUILD DEFAULT CONSTRAINTS
#
# State:
#
#     V_MIN <= v <= V_MAX
#
# We do NOT directly constrain theta.
#
# Position constraints can be added here if desired.
#
# Input:
#
#     A_MIN <= a <= A_MAX
#     -W_MAX <= omega <= W_MAX
#
# ============================================================


# ============================================================
# SOLVE WT-MPC
# ============================================================
def solve_wt_mpc(
    state: np.ndarray,
    goal: np.ndarray,
    nominal_states: np.ndarray,
    nominal_controls: np.ndarray,
    W_samples: np.ndarray,
    obstacles: np.ndarray,
    config: "WTMPCConfig",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    state = np.asarray(state, dtype=float)
    goal = np.asarray(goal, dtype=float)
    nominal_states = np.asarray(nominal_states, dtype=float)
    nominal_controls = np.asarray(nominal_controls, dtype=float)
    W_samples = np.asarray(W_samples, dtype=float)

    horizon = config.horizon
    n_samples = config.n_samples

    nx = 4
    nu = 2

    if state.shape != (nx,):
        raise ValueError(f"Expected state shape {(nx,)}, got {state.shape}.")

    if nominal_states.shape != (horizon + 1, nx):
        raise ValueError(f"nominal_states must have shape {(horizon + 1, nx)}, got {nominal_states.shape}.")

    if nominal_controls.shape != (horizon, nu):
        raise ValueError(f"nominal_controls must have shape {(horizon, nu)}, got {nominal_controls.shape}.")

    if W_samples.shape != (n_samples, horizon, nx):
        raise ValueError(f"W_samples must have shape {(n_samples, horizon, nx)}, got {W_samples.shape}.")

    # --------------------------------------------------------
    # Linearize nominal nonlinear trajectory
    # --------------------------------------------------------

    A_seq, B_seq, c_seq = linearize_nominal_trajectory(
        nominal_states,
        nominal_controls,
        config.dt,
    )

    # --------------------------------------------------------
    # TV-LQR
    # --------------------------------------------------------

    K_seq = compute_time_varying_lqr_gains(
        A_seq,
        B_seq,
        config.q_lqr,
        config.r_lqr,
        config.Qf,
    )

    # --------------------------------------------------------
    # Closed-loop error dynamics
    # --------------------------------------------------------

    Acl_seq = compute_closed_loop_matrices(
        A_seq,
        B_seq,
        K_seq,
    )

    # --------------------------------------------------------
    # Transition matrices
    # --------------------------------------------------------

    Phi = build_ltv_transition_products(Acl_seq)

    # --------------------------------------------------------
    # Sampled error trajectories
    # --------------------------------------------------------

    error_samples = propagate_error_samples(
        W_samples,
        Acl_seq,
    )

    # --------------------------------------------------------
    # State constraints
    # --------------------------------------------------------

    A_constraints = np.asarray(
        config.state_constraints.H,
        dtype=float,
    )

    b_constraints = -np.asarray(
        config.state_constraints.h,
        dtype=float,
    )

    # --------------------------------------------------------
    # Single-step disturbance support
    # --------------------------------------------------------

    F, g = build_box_support(config.w_max)

    n_F = F.shape[0]

    # --------------------------------------------------------
    # Decision variables
    # --------------------------------------------------------

    z = cp.Variable((nx, horizon + 1))

    v = cp.Variable((nu, horizon))

    # Note: zeta is no longer declared here as a 4D tensor.
    # It must be instantiated dynamically in the loop below.

    constraints = []

    # --------------------------------------------------------
    # Initial nominal state
    # --------------------------------------------------------

    constraints.append(z[:, 0] == state)

    # --------------------------------------------------------
    # Input tightening
    # --------------------------------------------------------

    input_tightening = compute_input_tightening(
        config.input_constraints.H,
        K_seq,
        Phi,
        config.w_max,
        horizon,
    )

    # --------------------------------------------------------
    # Recursive state tightening (Static)
    # --------------------------------------------------------

    recursive_tightening = compute_recursive_tightening(
        A_constraints,
        config.gamma,
        Phi,
        config.w_max,
        horizon,
    )

    # --------------------------------------------------------
    # Nominal dynamics
    # --------------------------------------------------------

    for k in range(1, horizon + 1):
        # 1. ENFORCE VEHICLE PHYSICS
        constraints.append(z[:, k] == A_seq[k - 1] @ z[:, k - 1] + B_seq[k - 1] @ v[:, k - 1] + c_seq[k - 1])

        # 2. ENFORCE STEERING AND ACCELERATION LIMITS
        constraints.append(
            config.input_constraints.H @ v[:, k - 1] <= config.input_constraints.h - input_tightening[k - 1]
        )

        D_k = build_D_k(Acl_seq, k)

        A_obs, b_obs = get_linearized_obstacle_constraints(nominal_states[k], obstacles)

        if A_obs.shape[0] > 0:
            A_k_total = np.vstack([A_constraints, A_obs])
            b_k_total = np.concatenate([b_constraints, b_obs])
            obs_tightening_all = compute_recursive_tightening(A_obs, config.gamma, Phi, config.w_max, horizon)
            tightening_k = np.concatenate([recursive_tightening[k], obs_tightening_all[k]])
        else:
            A_k_total = A_constraints
            b_k_total = b_constraints
            tightening_k = recursive_tightening[k]

        n_total_constraints = A_k_total.shape[0]

        # INSTANTIATE PER-TIMESTEP VARIABLES HERE
        tau_k = cp.Variable()
        lam_k = cp.Variable(nonneg=True)
        s_k = cp.Variable(n_samples)
        zeta_k = cp.Variable((n_samples, n_total_constraints, n_F), nonneg=True)

        add_dr_cvar_constraints(
            constraints,
            z[:, k],
            error_samples[:, k],
            A_k_total,
            b_k_total,
            F,
            g,
            D_k,
            config.gamma,
            config.epsilon,
            tau_k,  # Pass local step variables
            lam_k,
            s_k,
            zeta_k,
            tightening_k,
        )

    # --------------------------------------------------------
    # Objective
    # --------------------------------------------------------

    goal_heading = np.arctan2(
        goal[1] - state[1],
        goal[0] - state[0],
    )

    goal_state = np.array(
        [
            goal[0],
            goal[1],
            goal_heading,
            target_v,
        ]
    )

    cost = 0.0

    for k in range(horizon):
        state_error = z[:, k] - goal_state

        cost += cp.quad_form(
            state_error,
            config.Q,
        )

        cost += cp.quad_form(
            v[:, k],
            config.R,
        )

    cost += cp.quad_form(
        z[:, horizon] - goal_state,
        config.Qf,
    )

    # --------------------------------------------------------
    # Solve
    # --------------------------------------------------------

    problem = cp.Problem(
        cp.Minimize(cost),
        constraints,
    )

    try:
        problem.solve(
            solver=cp.CLARABEL,
            verbose=False,
        )

    except Exception as exc:
        print(f"CLARABEL failed. Falling back to SCS: {exc}")

        problem.solve(
            solver=cp.SCS,
            verbose=False,
        )

    if problem.status not in (
        cp.OPTIMAL,
        cp.OPTIMAL_INACCURATE,
    ):
        raise RuntimeError(f"WT-MPC solve failed. Status: {problem.status}")

    # --------------------------------------------------------
    # Extract solution
    # --------------------------------------------------------

    z_opt = np.asarray(
        z.value,
        dtype=float,
    ).T

    v_opt = np.asarray(
        v.value,
        dtype=float,
    ).T

    # --------------------------------------------------------
    # Actual feedback
    # --------------------------------------------------------

    e_0 = state - z_opt[0]

    u_opt = v_opt[0] + K_seq[0] @ e_0

    # --------------------------------------------------------
    # Clip only for numerical safety.
    # --------------------------------------------------------

    u_opt[0] = np.clip(
        u_opt[0],
        A_MIN,
        A_MAX,
    )

    u_opt[1] = np.clip(
        u_opt[1],
        W_MIN,
        W_MAX,
    )

    return (
        u_opt,
        z_opt,
        v_opt,
        K_seq,
        error_samples,
    )


# ============================================================
# COLLISION CHECK
# ============================================================


def collides_with_obstacles(
    state: np.ndarray,
    obstacles: np.ndarray,
) -> bool:

    if obstacles.shape[0] == 0:
        return False

    position = state[:2]

    distances = np.linalg.norm(
        obstacles[:, :2] - position[None, :],
        axis=1,
    )

    return bool(np.any(distances <= obstacles[:, 2]))


# ============================================================
# GOAL CHECK
# ============================================================


def goal_reached_check(
    state: np.ndarray,
    goal: np.ndarray,
    tolerance: float = WT_MPC_GOAL_TOLERANCE,
) -> bool:

    return bool(np.linalg.norm(state[:2] - goal[:2]) <= tolerance)


# ============================================================
# SHIFT NOMINAL TRAJECTORY
#
# We warm-start the next MPC iteration using the previous
# optimized nominal trajectory.
#
# Input:
#
#     z_plan: (horizon + 1, nx)
#     v_plan: (horizon, nu)
#
# Output:
#
#     nominal_states: (horizon + 1, nx)
#     nominal_controls: (horizon, nu)
# ============================================================


def shift_nominal_trajectory(
    z_plan: np.ndarray,
    v_plan: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    nominal_states = np.vstack(
        [
            z_plan[1:],
            z_plan[-1],
        ]
    )

    nominal_controls = np.vstack(
        [
            v_plan[1:],
            v_plan[-1],
        ]
    )

    return (
        nominal_states,
        nominal_controls,
    )


# ============================================================
# RUN WT-MPC SIMULATION
# ============================================================


def run_wt_mpc_simulation(
    alg_key,
    env_key,
    initial_state: np.ndarray,
    offline_dataset,
    obstacles: np.ndarray,
    config: WTMPCConfig,
    n_steps: int,
    goal_tolerance: float = WT_MPC_GOAL_TOLERANCE,
    verbose: bool = True,
    return_stats: bool = False,
    sensor_noise_scale: float = 0.0,
    sensor_noise_shape: float = SENSOR_NOISE_SHAPE,
):

    state = np.asarray(
        initial_state,
        dtype=float,
    ).copy()

    goal = np.asarray(
        target_pos,
        dtype=float,
    )

    if state.shape != (4,):
        raise ValueError(f"Expected initial_state shape (4,), got {state.shape}.")

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    states = [state.copy()]

    controls = []

    nominal_states_history = []

    nominal_controls_history = []

    disturbances = []

    time_history = []

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    hit_obstacle = False

    goal_reached = False

    infeasible = False

    time_to_goal = np.inf

    # --------------------------------------------------------
    # Initial nominal trajectory
    # --------------------------------------------------------

    nominal_states, nominal_controls = initialize_nominal_trajectory(
        state,
        goal,
        config.horizon,
    )

    _, true_disturbance_sequence = sample_coupled_true_process_sequence(
        env_key,
        offline_dataset,
        n_steps=n_steps,
        sequence_horizon=config.horizon,
        beta=TRUE_BETA,
        shift_direction=None,
        sequence_metric="l2",
    )

    # --------------------------------------------------------
    # Receding horizon loop
    # --------------------------------------------------------

    for t in range(n_steps):
        if goal_reached:
            break

        t0 = time.time()

        alg_key, sample_key, sensor_key = jax.random.split(alg_key, 3)

        if sensor_noise_scale > 0:
            sensor_noise = generate_skew_normal_noise(
                sensor_key, obstacles.shape, sensor_noise_scale, sensor_noise_shape
            )
            sensor_noise = np.clip(
                sensor_noise,
                -sensor_noise_scale * SENSOR_NOISE_CLIP,
                sensor_noise_scale * SENSOR_NOISE_CLIP,
            )
            sensed_obstacles = obstacles + sensor_noise
        else:
            sensed_obstacles = obstacles

        # ----------------------------------------------------
        # Sample N*T disturbances
        # ----------------------------------------------------

        W_flat = sample_empirical_disturbances(
            sample_key,
            offline_dataset,
            config.n_samples * config.horizon,
        )

        W_flat = np.asarray(W_flat)

        if W_flat.shape[-1] != DIST_DIM:
            raise ValueError(f"Expected disturbance dimension {DIST_DIM}, got {W_flat.shape[-1]}.")

        W_samples = W_flat.reshape(
            config.n_samples,
            config.horizon,
            DIST_DIM,
        )

        # ----------------------------------------------------
        # Solve WT-MPC
        # ----------------------------------------------------

        try:
            (
                u,
                z_plan,
                v_plan,
                K_seq,
                error_samples,
            ) = solve_wt_mpc(
                state,
                goal,
                nominal_states,
                nominal_controls,
                W_samples,
                sensed_obstacles,
                config,
            )

        except RuntimeError as exc:
            infeasible = True

            if verbose:
                print(f"WT-MPC failed at step {t}: {exc}")

            break

        computation_time = time.time() - t0
        time_history.append(computation_time)
        # ----------------------------------------------------
        # True disturbance realization
        # ----------------------------------------------------

        true_disturbance = true_disturbance_sequence[t]

        # ----------------------------------------------------
        # Apply actual control and true disturbance
        # ----------------------------------------------------

        next_state = np.asarray(
            dynamics_step(
                jnp.asarray(state),
                jnp.asarray(u),
                jnp.asarray(true_disturbance),
            )
        )

        # ----------------------------------------------------
        # Log
        # ----------------------------------------------------

        states.append(next_state.copy())

        controls.append(u.copy())

        nominal_states_history.append(z_plan.copy())

        nominal_controls_history.append(v_plan.copy())

        disturbances.append(true_disturbance.copy())

        if verbose:
            print(f"Step {t}: WT-MPC computation time = {computation_time:.3f} s, state = {next_state}, control = {u}")

        # ----------------------------------------------------
        # Collision
        # ----------------------------------------------------

        if collides_with_obstacles(
            next_state,
            obstacles,
        ):
            hit_obstacle = True

            if verbose:
                print(f"Collision detected at step {t}.")

            break

        # ----------------------------------------------------
        # Goal
        # ----------------------------------------------------

        if goal_reached_check(
            next_state,
            goal,
            goal_tolerance,
        ):
            goal_reached = True

            # Actual elapsed simulation time
            time_to_goal = (t + 1) * config.dt

            if verbose:
                print(f"Goal reached at step {t}.")

            break

        # ----------------------------------------------------
        # Update state
        # ----------------------------------------------------

        state = next_state

        # ----------------------------------------------------
        # Warm-start next nominal trajectory
        # ----------------------------------------------------

        (
            nominal_states,
            nominal_controls,
        ) = shift_nominal_trajectory(
            z_plan,
            v_plan,
        )

    # ========================================================
    # BUILD RESULT
    # ========================================================

    result = (
        np.asarray(states),
        np.asarray(controls),
    )
    if not return_stats:
        return result
    # ========================================================
    # BUILD STATS
    # ========================================================

    stats = {
        "hit_obstacle": bool(hit_obstacle),
        "infeasible": bool(infeasible),
        "goal_reached": bool(goal_reached),
        "time_to_goal": (float(time_to_goal) if goal_reached else float(np.inf)),
        "mean_computation_time": float(np.mean(time_history)) if time_history else 0.0,
    }

    return (
        *result,
        stats,
    )
