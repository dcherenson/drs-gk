"""Dynamics, policies, geometry, and disturbance sampling primitives."""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from config import (
    DT,
    V_MIN,
    V_MAX,
    A_MIN,
    A_MAX,
    W_MAX,
    K_P,
    K_THETA,
    DATASET_SIZE,
    NOMINAL_NOISE_MEAN,
    NOMINAL_NOISE_STD,
    OUTLIER_NOISE_MEAN,
    OUTLIER_NOISE_STD,
    OUTLIER_PROBABILITY,
    OBSTACLE_STYLE,
)

plt.switch_backend("Agg")


def generate_skew_normal_noise(key, shape, scale, alpha):
    """
    Generates zero-mean skew-normal noise.
    alpha: shape parameter
    scale: scale parameter (omega)
    """
    key_u0, key_u1 = jax.random.split(key)
    u0 = jax.random.normal(key_u0, shape)
    u1 = jax.random.normal(key_u1, shape)

    delta = alpha / jnp.sqrt(1.0 + alpha**2)
    z = delta * jnp.abs(u0) + jnp.sqrt(1.0 - delta**2) * u1

    mean_z = delta * jnp.sqrt(2.0 / jnp.pi)

    return scale * (z - mean_z)


@jax.jit
def dynamics_step(state, control, disturbance):
    px, py, theta, v = state[0], state[1], state[2], state[3]
    ua, uw = control[0], control[1]

    # Additive disturbances (state-dependent)
    xi_a, xi_w = disturbance[0], disturbance[1]

    # Multiplicative disturbances (control-dependent)
    eta_a, eta_w = disturbance[2], disturbance[3]

    # Continuous time derivatives
    px_dot = v * jnp.cos(theta)
    py_dot = v * jnp.sin(theta)
    theta_dot = (1.0 + eta_w) * uw + (v * xi_w)
    v_dot = (1.0 + eta_a) * ua + (v * xi_a)

    # Discrete time integration
    next_px = px + (px_dot * DT)
    next_py = py + (py_dot * DT)
    next_theta = theta + (theta_dot * DT)
    next_v = jnp.clip(v + (v_dot * DT), V_MIN, V_MAX)

    return jnp.array([next_px, next_py, next_theta, next_v])


@jax.jit
def nominal_policy(state, target_pos, target_v):
    px, py, theta, v = state[0], state[1], state[2], state[3]
    goal_x, goal_y = target_pos[0], target_pos[1]

    # Proportional velocity control
    ua = K_P * (target_v - v)

    # Heading control
    theta_des = jnp.arctan2(goal_y - py, goal_x - px)
    heading_error = theta_des - theta

    # Wrap heading error to [-pi, pi]
    heading_error = jnp.mod(heading_error + jnp.pi, 2 * jnp.pi) - jnp.pi
    uw = K_THETA * heading_error

    # Apply control saturation
    ua_clipped = jnp.clip(ua, A_MIN, A_MAX)
    uw_clipped = jnp.clip(uw, -W_MAX, W_MAX)

    return jnp.array([ua_clipped, uw_clipped])


@jax.jit
def backup_policy(state, obstacle_center, locked_turn_direction=0.0):
    px, py, theta, v = state[0], state[1], state[2], state[3]
    obstacle_x, obstacle_y = obstacle_center[0], obstacle_center[1]

    # Decelerate to minimum speed
    ua = jnp.where(v > V_MIN, A_MIN, 0.0)

    # Determine turn direction away from the active obstacle center
    rel_px = px - obstacle_x
    rel_py = py - obstacle_y
    turn_metric = rel_py * jnp.cos(theta) - rel_px * jnp.sin(theta)

    # Use jnp.sign to apply max yaw rate in the required direction
    # Add a small epsilon to prevent zero output if perfectly aligned
    computed_turn_direction = jnp.sign(turn_metric + 1e-6)

    actual_turn_direction = jnp.where(locked_turn_direction == 0.0, computed_turn_direction, locked_turn_direction)

    uw = actual_turn_direction * W_MAX

    # Apply control saturation
    ua_clipped = jnp.clip(ua, A_MIN, A_MAX)
    uw_clipped = jnp.clip(uw, -W_MAX, W_MAX)

    return jnp.array([ua_clipped, uw_clipped])


def format_obstacles(obstacles, allow_empty=False):
    obstacles_np = np.asarray(obstacles, dtype=float)
    if obstacles_np.ndim != 2 or obstacles_np.shape[1] != 3:
        raise ValueError("Obstacles must have shape (n_obstacles, 3) with rows [center_x, center_y, radius].")
    if obstacles_np.shape[0] == 0 and not allow_empty:
        raise ValueError("At least one obstacle is required.")
    return obstacles_np


def obstacle_clearances(state, obstacles):
    obstacles_np = format_obstacles(obstacles)
    position = np.asarray(state[:2], dtype=float)
    centers = obstacles_np[:, :2]
    radii = obstacles_np[:, 2]
    return np.linalg.norm(centers - position, axis=1) - radii


def closest_obstacle_index(state, obstacles):
    return int(np.argmin(obstacle_clearances(state, obstacles)))


def visible_obstacle_indices(state, obstacles, visual_radius):
    return np.where(obstacle_clearances(state, obstacles) <= visual_radius)[0]


def collides_with_any_obstacle(state, obstacles):
    return bool(np.any(obstacle_clearances(state, obstacles) <= 0.0))


def add_obstacles_to_axis(ax, obstacles):
    for obstacle_idx, (center_x, center_y, radius) in enumerate(format_obstacles(obstacles)):
        obstacle = plt.Circle(
            (center_x, center_y),
            radius,
            **OBSTACLE_STYLE,
            label="Obstacle" if obstacle_idx == 0 else "_nolegend_",
        )
        ax.add_patch(obstacle)


def closest_obstacle_center_jax(state, obstacle_centers, obstacle_radii):
    clearances = jnp.linalg.norm(obstacle_centers - state[:2], axis=1) - obstacle_radii
    return obstacle_centers[jnp.argmin(clearances)]


def generate_offline_dataset(key, dataset_size=DATASET_SIZE):
    """
    Generates a synthetic empirical dataset of disturbance residuals [xi_a, xi_w, eta_a, eta_w].
    """
    keys = jax.random.split(key, 9)
    key_nom_a, key_nom_w, key_nom_eta_a, key_nom_eta_w = keys[0:4]
    key_out_a, key_out_w, key_out_eta_a, key_out_eta_w = keys[4:8]
    key_mix = keys[8]

    # Mode 1: Nominal symmetric noise (80% probability)
    mu_nom = jnp.asarray(NOMINAL_NOISE_MEAN)
    sigma_nom = jnp.asarray(NOMINAL_NOISE_STD)
    nom_a = mu_nom[0] + sigma_nom[0] * jax.random.normal(key_nom_a, (dataset_size,))
    nom_w = mu_nom[1] + sigma_nom[1] * jax.random.normal(key_nom_w, (dataset_size,))
    nom_eta_a = mu_nom[2] + sigma_nom[2] * jax.random.normal(key_nom_eta_a, (dataset_size,))
    nom_eta_w = mu_nom[3] + sigma_nom[3] * jax.random.normal(key_nom_eta_w, (dataset_size,))
    nominal_samples = jnp.stack([nom_a, nom_w, nom_eta_a, nom_eta_w], axis=1)

    # Mode 2: Asymmetric outlier noise (20% probability)
    # Introduces biases in additive terms and systematic gain reductions in multiplicative terms
    mu_out = jnp.asarray(OUTLIER_NOISE_MEAN)
    sigma_out = jnp.asarray(OUTLIER_NOISE_STD)
    out_a = mu_out[0] + sigma_out[0] * jax.random.normal(key_out_a, (dataset_size,))
    out_w = mu_out[1] + sigma_out[1] * jax.random.normal(key_out_w, (dataset_size,))
    out_eta_a = mu_out[2] + sigma_out[2] * jax.random.normal(key_out_eta_a, (dataset_size,))
    out_eta_w = mu_out[3] + sigma_out[3] * jax.random.normal(key_out_eta_w, (dataset_size,))
    outlier_samples = jnp.stack([out_a, out_w, out_eta_a, out_eta_w], axis=1)

    # Construct the mixture
    mixture_mask = jax.random.bernoulli(key_mix, p=OUTLIER_PROBABILITY, shape=(dataset_size, 1))
    empirical_dataset = jnp.where(mixture_mask, outlier_samples, nominal_samples)

    return empirical_dataset


@jax.jit(static_argnames=["n_samples"])
def sample_empirical_disturbances(key, dataset, n_samples):
    """
    Draws N empirical samples uniformly with replacement from the historical dataset.
    n_samples must be static for JIT compilation to trace the output shape.
    """
    dataset_size = dataset.shape[0]
    indices = jax.random.choice(key, jnp.arange(dataset_size), shape=(n_samples,), replace=True)
    return dataset[indices]


def _validated_shift_direction(shift_direction, dimension):
    """Return a unit-norm shift direction, using a random default."""
    if shift_direction is None:
        direction = np.random.normal(size=dimension)
        while np.linalg.norm(direction) == 0.0:
            direction = np.random.normal(size=dimension)
    else:
        direction = np.asarray(shift_direction, dtype=float)

    if direction.ndim != 1 or direction.shape[0] != dimension:
        raise ValueError(f"shift_direction must have shape ({dimension},).")
    if not np.all(np.isfinite(direction)):
        raise ValueError("shift_direction must contain only finite values.")

    direction_norm = np.linalg.norm(direction)
    if direction_norm == 0.0:
        raise ValueError("shift_direction must be nonzero.")
    return jnp.asarray(direction / direction_norm)


def _validated_beta(beta):
    beta = float(beta)
    if not np.isfinite(beta) or beta < 0.0:
        raise ValueError("beta must be a finite, nonnegative scalar.")
    return beta


def _validated_sequence_metric(sequence_metric):
    if sequence_metric not in {"max", "l2"}:
        raise ValueError("sequence_metric must be either 'max' or 'l2'.")
    return sequence_metric


def shift_empirical_dataset(dataset, beta, shift_direction=None):
    """
    Construct one member of an infinity-Wasserstein ambiguity set.

    If X has the empirical nominal distribution P, this returns the empirical
    distribution of Y = X + beta * u, where u is a unit-norm direction.  The
    coupling (X, Y) therefore has ||X - Y||_2 <= beta, so
    W_infinity(P, Law(Y)) <= beta.  The ambiguity set does not identify a
    unique true distribution; ``shift_direction`` selects this particular
    member.  By default the direction is the normalized all-ones vector.
    """
    dataset = jnp.asarray(dataset)
    if dataset.ndim != 2 or dataset.shape[0] == 0:
        raise ValueError("dataset must have shape (n_samples, n_dimensions) and be nonempty.")

    beta = _validated_beta(beta)
    direction = _validated_shift_direction(shift_direction, dataset.shape[1])
    return dataset + jnp.asarray(beta, dtype=dataset.dtype) * direction


def sample_coupled_true_disturbances(
    key,
    dataset,
    n_samples,
    beta,
    shift_direction=None,
):
    """
    Sample nominal and true disturbances under an explicit transport coupling.

    Each returned pair is (x, y) with x sampled from the empirical nominal
    dataset and y = x + beta*u sampled from the shifted true distribution.
    Keeping this coupling is useful when checking the Wasserstein guarantee;
    independently sampled nominal and true values need not be close sample by
    sample even when their distributions are close in W_infinity.
    """
    dataset = jnp.asarray(dataset)
    if dataset.ndim != 2 or dataset.shape[0] == 0:
        raise ValueError("dataset must have shape (n_samples, n_dimensions) and be nonempty.")
    if int(n_samples) < 1:
        raise ValueError("n_samples must be positive.")

    beta = _validated_beta(beta)
    direction = _validated_shift_direction(shift_direction, dataset.shape[1])
    nominal_samples = sample_empirical_disturbances(key, dataset, int(n_samples))
    true_samples = nominal_samples + jnp.asarray(beta, dtype=dataset.dtype) * direction
    return nominal_samples, true_samples


def sample_true_disturbances(key, dataset, n_samples, beta, shift_direction=None):
    """
    Sample from the shifted true empirical distribution Q_beta.

    Q_beta is the pushforward of the nominal empirical distribution under
    x -> x + beta*u.  This one-step sampler is suitable for the closed-loop
    simulation.  For a coupled T-step sample, use
    ``sample_coupled_true_disturbance_sequences`` instead.
    """
    _, true_samples = sample_coupled_true_disturbances(
        key,
        dataset,
        n_samples,
        beta,
        shift_direction=shift_direction,
    )
    return true_samples


def sample_coupled_true_disturbance_sequences(
    key,
    dataset,
    n_sequences,
    horizon,
    beta,
    shift_direction=None,
    sequence_metric="l2",
):
    """
    Sample coupled nominal/true disturbance sequences.

    The sequence ground metrics are:

    * ``max``: d_T(W, W') = max_t ||w_t - w'_t||_2.  Every T-step sequence
      then has transport cost at most ``beta``.
    * ``l2``: d_T(W, W') is the Euclidean norm after flattening the sequence.
      The per-step displacement is scaled by 1/sqrt(T), so the flattened
      transport cost is at most ``beta``.

    Thus the returned coupling certifies
    W_infinity(P^T, Q_{beta,T}^T) <= beta for the selected sequence metric,
    where Q_{beta,T} is the shifted law selected by this sampler.
    """
    dataset = jnp.asarray(dataset)
    if dataset.ndim != 2 or dataset.shape[0] == 0:
        raise ValueError("dataset must have shape (n_samples, n_dimensions) and be nonempty.")
    if int(n_sequences) < 1 or int(horizon) < 1:
        raise ValueError("n_sequences and horizon must be positive.")

    beta = _validated_beta(beta)
    sequence_metric = _validated_sequence_metric(sequence_metric)
    direction = _validated_shift_direction(shift_direction, dataset.shape[1])

    n_sequences = int(n_sequences)
    horizon = int(horizon)
    nominal_samples = sample_empirical_disturbances(
        key,
        dataset,
        n_sequences * horizon,
    ).reshape((n_sequences, horizon, dataset.shape[1]))

    step_scale = 1.0 if sequence_metric == "max" else 1.0 / np.sqrt(horizon)
    true_samples = (
        nominal_samples
        + jnp.asarray(
            beta * step_scale,
            dtype=dataset.dtype,
        )
        * direction
    )
    return nominal_samples, true_samples


def sample_coupled_true_process_sequence(
    key,
    dataset,
    n_steps,
    sequence_horizon,
    beta,
    shift_direction=None,
    sequence_metric="l2",
):
    """
    Sample a process sequence whose every ``sequence_horizon``-step window is
    coupled to a nominal window within ``beta``.

    This is used when the simulator runs for more than one DRS-GK horizon.
    For the flattened Euclidean sequence norm, each process-noise sample is
    shifted by beta/sqrt(sequence_horizon). Consequently, every contiguous
    window of ``sequence_horizon`` samples has total displacement at most
    beta, even though the entire longer simulation sequence may have a larger
    displacement.
    """
    dataset = jnp.asarray(dataset)
    if dataset.ndim != 2 or dataset.shape[0] == 0:
        raise ValueError("dataset must have shape (n_samples, n_dimensions) and be nonempty.")
    if int(n_steps) < 1 or int(sequence_horizon) < 1:
        raise ValueError("n_steps and sequence_horizon must be positive.")

    beta = _validated_beta(beta)
    sequence_metric = _validated_sequence_metric(sequence_metric)
    direction = _validated_shift_direction(shift_direction, dataset.shape[1])
    n_steps = int(n_steps)
    sequence_horizon = int(sequence_horizon)

    nominal_samples = sample_empirical_disturbances(key, dataset, n_steps)
    step_scale = 1.0 if sequence_metric == "max" else 1.0 / np.sqrt(sequence_horizon)
    true_samples = (
        nominal_samples
        + jnp.asarray(
            beta * step_scale,
            dtype=dataset.dtype,
        )
        * direction
    )
    return nominal_samples, true_samples


def sequence_transport_displacement(nominal_sequences, true_sequences, sequence_metric="l2"):
    """Compute the transport cost of paired sequences under the selected metric."""
    sequence_metric = _validated_sequence_metric(sequence_metric)
    nominal_sequences = jnp.asarray(nominal_sequences)
    true_sequences = jnp.asarray(true_sequences)
    if nominal_sequences.shape != true_sequences.shape or nominal_sequences.ndim != 3:
        raise ValueError("paired sequences must both have shape (n_sequences, horizon, dimension).")

    displacement = true_sequences - nominal_sequences
    per_step_norm = jnp.linalg.norm(displacement, axis=-1)
    if sequence_metric == "max":
        return jnp.max(per_step_norm, axis=-1)
    return jnp.linalg.norm(displacement.reshape((displacement.shape[0], -1)), axis=-1)


@jax.jit(static_argnames=["nominal_policy_fn"])
def rollout_with_switch(state, s, W, nominal_policy_fn, obstacle_centers, obstacle_radii):
    """
    Simulates a single rollout under a fixed switching time and returns all states.
    """

    def scan_step(carry, inputs):
        current_state, locked_direction = carry
        t, w_t = inputs

        current_closest = closest_obstacle_center_jax(current_state, obstacle_centers, obstacle_radii)

        rel_px = current_state[0] - current_closest[0]
        rel_py = current_state[1] - current_closest[1]
        turn_metric = rel_py * jnp.cos(current_state[2]) - rel_px * jnp.sin(current_state[2])
        computed_dir = jnp.sign(turn_metric + 1e-6)

        lock_time = jnp.maximum(0, s)
        new_locked_direction = jnp.where(t == lock_time, computed_dir, locked_direction)

        u = jax.lax.cond(
            t < s,
            lambda _: nominal_policy_fn(current_state),
            lambda _: backup_policy(current_state, current_closest, new_locked_direction),
            operand=None,
        )
        next_state = dynamics_step(current_state, u, w_t)
        return (next_state, new_locked_direction), next_state

    time_indices = jnp.arange(W.shape[0])
    initial_locked_direction = jnp.array(0.0)
    _, states = jax.lax.scan(scan_step, (state, initial_locked_direction), (time_indices, W))
    return jnp.concatenate([state[jnp.newaxis, :], states], axis=0)


@jax.jit(static_argnames=["nominal_policy_fn"])
def rollout_bundle(state, s, W_samples, nominal_policy_fn, obstacle_centers, obstacle_radii):
    """
    Vectorized rollout helper used for visualization.
    """
    return jax.vmap(rollout_with_switch, in_axes=(None, None, 0, None, None, None))(
        state,
        s,
        W_samples,
        nominal_policy_fn,
        obstacle_centers,
        obstacle_radii,
    )
