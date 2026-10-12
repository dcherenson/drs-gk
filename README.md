# Distributionally Robust Stochastic Gatekeeper

This repository contains a comparison of five safety filters for a variable-speed Dubins vehicle: DRS-GK, deterministic Gatekeeper, SMPS, DR-CBF, and WT-MPC. The simulator uses nonlinear dynamics, a finite empirical disturbance data set, obstacle sensing noise, and a Wasserstein distribution-shift model. The default comparison runs all five controllers on the same scenario and writes figures, animations, and metrics to `output/`.

The method is described in [“Distributionally Robust Safety Under Arbitrary Uncertainties: A Safety Filtering Approach”](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=11704677), published in **IEEE Robotics and Automation Letters (RA-L)** by Daniel M. Cherenson\*, Haejoon Lee\*, Taekyung Kim, and Dimitra Panagou (2026).

\* Equal contribution.

## Installation

Python 3.13 is the minimum supported interpreter. The baseline installation uses JAX on CPU and includes:

| Package | Minimum version |
| --- | ---: |
| `jax` | 0.10.2 |
| `numpy` | 2.4.4 |
| `matplotlib` | 3.10.8 |
| `scipy` | 1.17 |
| `cvxpy` | 1.7.2 |
| `pillow` | 12 |

The canonical project definition is [`pyproject.toml`](pyproject.toml), built with setuptools. The checked-in [`uv.lock`](uv.lock) has been tested with a fresh `uv sync` on Python 3.13 (JAX 0.11.2 and CVXPY 1.9.3). Using [uv](https://docs.astral.sh/uv/) is the recommended setup. If `uv` is not installed, install it from the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) or run `python -m pip install uv`.

```bash
git clone https://github.com/dcherenson/drs-gk.git
cd drs-gk
uv sync
uv run python main.py --quick
```

The equivalent pip setup, after cloning the repository and changing into `drs-gk`, is:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py --quick
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`.

For an NVIDIA Linux installation, JAX's optional CUDA 13 wheel can be installed with pip instead of the CPU build. Follow the [official JAX installation instructions](https://docs.jax.dev/en/latest/installation.html) for the driver and CUDA requirements. CVXPY installs its supported solver dependencies automatically; see the [official CVXPY installation guide](https://www.cvxpy.org/install/index.html). The code uses Clarabel and falls back to SCS, so no separately licensed solver is required. Pillow is sufficient for GIF export; `ffmpeg` is not required for the GIF examples in this repository.

## Running the comparison

The main entry point is `main.py`. Useful commands are:

```bash
# Full comparison with the nominal settings.
python main.py

# DRS-GK-only demo that runs until the goal is reached (up to 600 steps).
python main.py --quick

# Run only selected controllers with the nominal settings.
python main.py --controllers drs-gk dr-cbf --no-animation

# Run the configured experiment modes.
python main.py --experiment initial-states --quick --no-animation
python main.py --experiment sweep --quick
python main.py --experiment eps-beta --quick
python main.py --experiment monte-carlo --quick
```

The runner also accepts `--seed`, `--output-dir`, and `--verbose`. Controller names are `drs-gk`, `gatekeeper`, `smps`, `dr-cbf`, and `wt-mpc`.

The default output directory is `output/` (ignored by the repository). A comparison run can produce `comparison.png`, `comparison.gif`, and `metrics.csv`; experiment modes add their CSV files and plots there. To summarize and plot a completed hyperparameter sweep, run:

```bash
python -m experiments.analyze_hyperparameter_sweep output/hyperparam_sweep_results_gk.csv
```

The experiment implementations are organized in [`experiments/sweeps.py`](experiments/sweeps.py), [`experiments/monte_carlo_analysis.py`](experiments/monte_carlo_analysis.py), and [`experiments/analyze_hyperparameter_sweep.py`](experiments/analyze_hyperparameter_sweep.py).

## Configuration

All tunable experiment settings live in [`config.py`](config.py). The modules in `helper.py` and `pcis.py` provide computational primitives and set construction; they are not a second configuration interface.

The nominal settings retain the original experiment scale:

| Setting | Value |
| --- | ---: |
| Integration step `DT` | `0.05` |
| Speed clip | `[9, 20]` |
| Input limits `(a, omega)` | `[-5, 5]` and `[-pi/4, pi/4]` |
| Simulation steps `N_STEPS` | `300` |
| DRS-GK `(M, T, N)` | `(50, 75, 1000)` |
| DRS-GK `(epsilon, delta)` | `(0.05, 0.1)` |
| True-process Wasserstein radius `TRUE_BETA` | `0.01` |
| DRS-GK `K_trials` | `100` |
| PCIS `(alpha, num_samples)` | `(0.01, 1000)` |
| DR-CBF `(alpha1, epsilon, sample_count)` | `(2.0, 0.1, 1000)` |
| WT-MPC `(horizon, n_samples)` | `(20, 20)` |

`--quick` runs only DRS-GK, with repeated safety checks and up to 600 simulation steps. The run stops when the goal is reached, a collision occurs, or the step limit is reached. The demo uses `(M,T,N)=(40,75,300)` and 100 PCIS samples, and writes the full trajectory and animation to `output/`. Use `--steps` to override the step limit. To compare controllers, omit `--quick`.

For experiment modes, `--quick` retains reduced settings: DRS-GK `(M,T,N)=(10,20,100)`, `K_trials=3`, and the Monte Carlo profile `(100 runs, 50-step horizon)`. The quick `eps-beta` grid uses `epsilon` in `{0.02, 0.05}` and `beta` in `{0.01, 0.05}`; the initial-state demo samples 3 starts. Use the nominal configuration for research runs.

## Model and uncertainty

The state and input are

$$
x = [p_x,p_y,\theta,v]^\top,\qquad u=[a,\omega]^\top,
$$

and the disturbance vector is

$$
w=[\xi_a,\xi_\omega,\eta_a,\eta_\omega]^\top.
$$

The continuous-time model used by the simulator is

$$
\begin{aligned}
\dot p_x &= v\cos\theta, &\quad \dot p_y &= v\sin\theta,\\
\dot\theta &= (1+\eta_\omega)\omega + v\xi_\omega,
&\quad \dot v &= (1+\eta_a)a + v\xi_a.
\end{aligned}
$$

Forward Euler integration with `DT=0.05` gives

$$
\begin{aligned}
p_{x,k+1} &= p_{x,k}+\Delta t\,v_k\cos\theta_k,\\
p_{y,k+1} &= p_{y,k}+\Delta t\,v_k\sin\theta_k,\\
\theta_{k+1} &= \theta_k+\Delta t\big((1+\eta_{\omega,k})\omega_k+v_k\xi_{\omega,k}\big),\\
v_{k+1} &= \mathrm{clip}\!(v_k+\Delta t\big((1+\eta_{a,k})a_k+v_k\xi_{a,k}\big),9,20).
\end{aligned}
$$

The nominal goal policy uses proportional velocity and heading errors, with gains `K_P=1` and `K_THETA=2`, followed by the input limits. The shared backup policy brakes toward the minimum speed and turns at the maximum yaw rate away from the closest sensed obstacle.

The offline disturbance data set is sampled from the two-component Gaussian mixture

$$
w\sim 0.8\,\mathcal N(\mu_{\rm nom},\mathrm{diag}(\sigma_{\rm nom}^2))
 +0.2\,\mathcal N(\mu_{\rm out},\mathrm{diag}(\sigma_{\rm out}^2)),
$$

with

```text
mu_nom  = [ 0.00,  0.00,  0.00, 0.00]   sigma_nom  = [0.05, 0.02, 0.03, 0.03]
mu_out  = [-0.15,  0.08, -0.10, 0.05]   sigma_out  = [0.02, 0.01, 0.05, 0.05]
```

The default data set has 5,000 samples. Predictive samples are drawn uniformly with replacement from its empirical distribution $\hat P$. When enabled, obstacle sensor noise is mean-centered skew-normal noise with scale 1, shape 5, and clipping at three scales for each sensed obstacle center and radius.

Distribution shift is represented by the infinity-Wasserstein ambiguity set

$$
\mathcal M_\beta(\hat P)=\{Q:W_\infty(Q,\hat P)\le\beta\}.
$$

The shifted sampler uses a unit direction $d$, $\lVert d\rVert_2=1$, so a one-step coupled pair has

$$
w^{\rm true}=w+\beta d.
$$

For a `T`-step sequence under the flattened Euclidean metric, the per-step displacement is $\beta/\sqrt{T}$:

$$
\mathrm{vec}(W^{\rm true})
 =\mathrm{vec}(W)+\frac{\beta}{\sqrt{T}}[d,\ldots,d],
\qquad
\lVert\mathrm{vec}(W^{\rm true})-\mathrm{vec}(W)\rVert_2=\beta.
$$

The coupling is used to generate the realized process sequence; predictive rollouts continue to use empirical samples unless a controller description below says otherwise.

## Controller equations

### DRS-GK

DRS-GK evaluates candidate switch times $s\in\{0,\ldots,M-1\}$. A rollout uses the nominal policy before $s$ and the backup policy from $s$ onward. Its safety margin is

$$
H_T(x,s,W)=\min\{\min_{1\le k\le T} c(x_k),\ h_{\rm PCIS}(x_T)\},
\qquad
c(x)=\min_i(\lVert p-c_i\rVert_2-R_i),
$$

where $h_{\rm PCIS}$ is the terminal margin for the precomputed probabilistic controlled-invariant set. For each candidate, the implementation evaluates $H_T$ and its gradient on the sampled rollout set and computes the empirical quantity

$$
L_H=\max_i\lVert\nabla_W H_T(x,s,W_i)\rVert_2.
$$

The distribution-shift buffer used by the implementation is $b_\beta=L_H\beta^{1/p}$ with `p=1` in the runner. A sampled rollout counts as a buffered failure when $H_T\le b_\beta$. If $k_s$ of `N` samples fail, the candidate's Clopper–Pearson upper bound is

$$
\bar q_s=\mathrm{Beta}^{-1}(1-\rho;k_s+1,N-k_s),
\qquad
\rho=1-(1-\delta)^{1/M}.
$$

When all `N` samples fail, the upper bound is set to $\bar q_s=1$.

The PCIS failure allowance is `alpha=0.01`; therefore DRS-GK accepts candidates with

$$
\bar q_s\le\frac{\varepsilon-\alpha}{1-\alpha}
$$

and selects the latest accepted switch time to preserve nominal-policy use. `L_H` is the maximum over the evaluated samples and is an empirical rollout quantity; it is not treated as a certified global Lipschitz constant.

### Deterministic Gatekeeper

The deterministic Gatekeeper uses the same candidate switch-time idea and shared nominal/backup policies. It rolls out each candidate with zero predictive disturbance. If $c_{\rm orbit}(x_T)$ is the loiter-center position computed from the terminal state, its implemented margin is:

$$
H_T^{\rm det}(x,s)=\min\{
\min_{0\le k<T,\,i}\big(\lVert p_k-c_i\rVert_2-R_i\big),
\min_i\big(\lVert c_{\rm orbit}(x_T)-c_i\rVert_2-v_T/W_{\max}-R_i\big)
\}.
$$

It selects the largest candidate with $H_T^{\rm det}\ge0$. The closed-loop simulator still applies the configured realized disturbance sequence when propagating the selected action.

### SMPS

SMPS performs a recoverability test at the current state. Its statistical sample count is derived from

$$
\varepsilon_t=\frac{6\varepsilon}{(t+1)^2\pi^2},\qquad
\delta_t=\frac{6\delta}{(t+1)^2\pi^2},\qquad
K_t=\lceil\frac{\log(1/\delta_t)}{\log(1/(1-\varepsilon_t))}\rceil.
$$

In `execute_smps`, the current-step test evaluates this helper with `t=0` and samples `K` empirical disturbance trajectories. It uses the nominal action only when every sampled rollout has $H_T\ge0$; otherwise it applies the shared backup action for that step. SMPS makes this decision again at the next step rather than retaining a future switch time.

### DR-CBF

For the selected obstacle, let $r=p-c=(r_x,r_y)$ and

$$
h(x)=r_x^2+r_y^2-R^2,
\quad A=2(r_x\cos\theta+r_y\sin\theta),
\quad B=2v(r_y\cos\theta-r_x\sin\theta).
$$

The higher-order barrier terms in the solver are

$$
h_e=Av+\alpha_1h,
\qquad C=2v^2+\alpha_1Av.
$$

For a disturbance sample $w_i$, the implemented barrier expression is

$$
\begin{aligned}
\mathrm{CBC}(u,w_i)={}&Aa+B\omega+Av\xi_{a,i}+Bv\xi_{\omega,i}\\
&+Aa\eta_{a,i}+B\omega\eta_{\omega,i}+C+\alpha_2h_e,
\end{aligned}
$$

with $G_i=-\mathrm{CBC}(u,w_i)$. The convex filter minimizes

$$
\min_{u,y,t,s,\alpha_2}\;y+|\alpha_2|,
\qquad \alpha_2\ge0,
$$

subject to

$$
\beta L+\frac1N\sum_i s_i-t\varepsilon\le0,
\quad s_i\ge G_i+t,
\quad s_i\ge0,
\quad y+1\ge\lVert[2(u-u_{\rm nom}),y-1]\rVert_2,
$$

where $L=\lVert[Av,Bv,Aa,B\omega]\rVert_2$, plus acceleration and yaw-rate bounds. Thus the second barrier gain is a nonnegative optimization variable in the objective, while `alpha1`, `epsilon`, `beta`, and the sample count remain configuration values.

### WT-MPC

WT-MPC linearizes a nominal trajectory and uses time-varying LQR feedback:

$$
x_{k+1}\approx A_kx_k+B_ku_k+c_k,
\qquad
e_{k+1}=(A_k+B_kK_k)e_k+w_k,
\qquad
u_k=\nu_k+K_ke_k.
$$

For its internal tube model, each sampled disturbance is treated as an additive four-dimensional state error. The support set is the box

$$
\mathcal W=\{w:-w_{\max}\le w\le w_{\max}\},
\qquad w_{\max}=[0.01,0.01,0.01,0.01].
$$

At each receding-horizon solve, sampled errors, support-based tightening, and the distributionally robust CVaR constraint

$$
\sup_{Q\in\mathcal B_W(\hat P,\varepsilon_{\rm MPC})}
\mathrm{CVaR}_{1-\gamma}\![\max_j(a_j^\top x_k+b_j)]\le0
$$

are combined with the linearized dynamics. The objective is the quadratic tracking cost

$$
\sum_{k=0}^{H-1}(\lVert z_k-x_g\rVert_Q^2+\lVert\nu_k\rVert_R^2)
 +\lVert z_H-x_g\rVert_{Q_f}^2.
$$

Only the first feedback action is applied; the nominal trajectory is shifted and the problem is solved again at the next step.

## Citation

```bibtex
@article{cherenson2026drs_gatekeeper,
  author  = {Cherenson, Daniel M. and Lee, Haejoon and Kim, Taekyung and Panagou, Dimitra},
  title   = {{Distributionally Robust Safety Under Arbitrary Uncertainties: A Safety Filtering Approach}},
  journal = {IEEE Robotics and Automation Letters},
  year    = {2026},
  volume  = {11},
  number  = {11},
  pages   = {13199--13206},
  doi     = {10.1109/LRA.2026.3737038},
  url     = {https://ieeexplore.ieee.org/document/11704677}
}
```
