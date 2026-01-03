import os
import json
import glob

import numpy as np
from fipy import CellVariable, Grid2D, TransientTerm, DiffusionTerm
from fipy.tools import numerix

# ============================================================
# Basic constants (consistent with your linear-aniso generator)
# ============================================================

DAY_SEC = 24.0 * 3600.0
MM_TO_M = 1e-3

# Make sure these match your generator:
D28x = 1e-12       # [m^2/s]
N_POLY = 2.0       # exponent in D(φ) = D28 * (1 + 5000 a φ^N_POLY)


# ============================================================
# 1. Linear anisotropic diffusion PDE (same as ML setup)
# ============================================================

def simulate_final_phi_anisotropic(params, ax_steps, ay_steps, nx=64, ny=64):
    """
    Re-run the anisotropic diffusion PDE with given stair-step a_x(t), a_y(t).

    params: dict loaded from params.json
      must contain width_mm, height_mm, c_boundary, c_init, c_other, time_days
    ax_steps: 1D array-like of length K, scalar a_x,k
    ay_steps: 1D array-like of length K, scalar a_y,k
    """
    width_mm   = params["width_mm"]
    height_mm  = params["height_mm"]
    c_boundary = params["c_boundary"]
    c_init     = params["c_init"]
    c_other    = params["c_other"]
    time_days  = params["time_days"]

    ax_steps = np.asarray(ax_steps, dtype=float)
    ay_steps = np.asarray(ay_steps, dtype=float)
    assert ax_steps.shape == ay_steps.shape
    num_steps = len(ax_steps)

    # --- Mesh ---
    Lx = width_mm * MM_TO_M
    Ly = height_mm * MM_TO_M

    dx = Lx / nx
    dy = Ly / ny

    mesh = Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)

    # --- Variable: concentration φ ---
    phi = CellVariable(
        name="concentration",
        mesh=mesh,
        value=c_init,
        hasOld=1,
    )

    # --- Boundary conditions ---
    X, Y = mesh.faceCenters
    exterior_faces = mesh.exteriorFaces

    y_min = (height_mm / 4.0) * MM_TO_M
    y_max = (3.0 * height_mm / 4.0) * MM_TO_M

    exposed_faces = mesh.facesLeft & (Y >= y_min) & (Y <= y_max)
    other_boundary_faces = exterior_faces & (~exposed_faces)

    phi.constrain(c_boundary, exposed_faces)
    phi.constrain(c_other, other_boundary_faces)

    # --- Time-stepping ---
    t = 0.0
    t_final = float(time_days) * DAY_SEC

    t_edges = numerix.linspace(0.0, t_final, num_steps + 1)

    for k in range(num_steps):
        interval_start = float(t_edges[k])
        interval_end   = float(t_edges[k + 1])

        ax_t = float(ax_steps[k])
        ay_t = float(ay_steps[k])

        # Stability-based dt using worst of ax, ay
        a_max = max(abs(ax_t), abs(ay_t))
        Dmax_ref = D28x * (1.0 + a_max)
        Dmax_ref = max(Dmax_ref, 1e-20)
        dt_base  = 0.25 * min(dx, dy)**2 / Dmax_ref

        while t < interval_end - 1e-12:
            dt = min(dt_base, interval_end - t)

            # Anisotropic, nonlinear D(φ)
            Dx_phi = D28x * (1.0 + 5000.0 * ax_t * phi**N_POLY)
            Dy_phi = D28x * (1.0 + 5000.0 * ay_t * phi**N_POLY)

            gamma = CellVariable(
                mesh=mesh,
                rank=2,
                value=((Dx_phi, 0 * phi),
                       (0 * phi, Dy_phi))
            )

            eqn = TransientTerm() == DiffusionTerm(coeff=gamma)

            phi.updateOld()
            TOL = 1e-2
            MAX_SWEEPS = 50

            res = 1.0
            for _ in range(MAX_SWEEPS):
                res = eqn.sweep(var=phi, dt=dt)
                if res <= TOL:
                    break

            t += dt

    # final φ as numpy array [ny, nx]
    phi_array = numerix.reshape(phi.value, (ny, nx))
    phi_np = np.array(phi_array)
    return phi_np


# ============================================================
# 2. Loss helper: Concentration MSE for [a_x(t), a_y(t)]
# ============================================================

def concentration_mse_for_profile(params, ax_steps, ay_steps, phi_true,
                                  nx_sim=64, ny_sim=64):
    """
    Given parameter profiles a_x(t), a_y(t) and ground-truth final field phi_true,
    simulate the PDE and return the concentration MSE.
    """
    phi_hat = simulate_final_phi_anisotropic(
        params, ax_steps, ay_steps, nx=nx_sim, ny=ny_sim
    )
    mse = float(np.mean((phi_hat - phi_true) ** 2))
    return mse


# ============================================================
# 3. Static anisotropic baseline
#    a_x(t) = const, a_y(t) = const
# ============================================================

def calibrate_static_anisotropic(
    params,
    phi_true,
    K,
    theta_min=1.0,
    theta_max=10.0,
    nx_sim=64,
    ny_sim=64,
    num_iters=10,
    lr=0.05,
    grad_eps=1e-2,
):
    """
    Static anisotropic baseline:

        a_x(t) = a_x_const,   a_y(t) = a_y_const  for all time intervals.

    Optimize (a_x_const, a_y_const) via gradient descent with finite-difference
    gradients, minimizing concentration MSE at final time.
    """

    def loss_fn(ax_const, ay_const):
        ax_c = float(np.clip(ax_const, theta_min, theta_max))
        ay_c = float(np.clip(ay_const, theta_min, theta_max))
        ax_vec = np.full(K, ax_c, dtype=float)
        ay_vec = np.full(K, ay_c, dtype=float)
        return concentration_mse_for_profile(
            params, ax_vec, ay_vec, phi_true, nx_sim, ny_sim
        )

    # Initialize at mid-range
    ax = 0.5 * (theta_min + theta_max)
    ay = 0.5 * (theta_min + theta_max)

    best_ax, best_ay = ax, ay
    best_loss = loss_fn(ax, ay)

    for it in range(num_iters):
        # Finite-difference grads for ax, ay (central difference)
        loss_center = loss_fn(ax, ay)

        # grad wrt ax
        loss_ax_plus  = loss_fn(ax + grad_eps, ay)
        loss_ax_minus = loss_fn(ax - grad_eps, ay)
        grad_ax = (loss_ax_plus - loss_ax_minus) / (2.0 * grad_eps)

        # grad wrt ay
        loss_ay_plus  = loss_fn(ax, ay + grad_eps)
        loss_ay_minus = loss_fn(ax, ay - grad_eps)
        grad_ay = (loss_ay_plus - loss_ay_minus) / (2.0 * grad_eps)

        # Gradient descent update
        ax = ax - lr * grad_ax
        ay = ay - lr * grad_ay

        # Clamp both into [theta_min, theta_max]
        ax = float(np.clip(ax, theta_min, theta_max))
        ay = float(np.clip(ay, theta_min, theta_max))

        loss_now = loss_fn(ax, ay)
        if loss_now < best_loss:
            best_loss = loss_now
            best_ax, best_ay = ax, ay

    return (best_ax, best_ay), best_loss


# ============================================================
# 4. Linear anisotropic baseline
#    a_x,k = ax0 + sx * τ_k,  a_y,k = ay0 + sy * τ_k
# ============================================================

def build_linear_aniso_profiles(
    ax0, sx, ay0, sy, K, theta_min=1.0, theta_max=10.0
):
    """
    Construct linear-in-time profiles for a_x(t), a_y(t) over K intervals:

        τ_k = k / (K - 1),  k = 0,...,K-1
        a_x,k = ax0 + sx * τ_k
        a_y,k = ay0 + sy * τ_k

    Then clamp each to [theta_min, theta_max].
    """
    if K == 1:
        t_norm = np.array([0.0], dtype=float)
    else:
        t_norm = np.linspace(0.0, 1.0, K, dtype=float)

    ax_vec = ax0 + sx * t_norm
    ay_vec = ay0 + sy * t_norm

    ax_vec = np.clip(ax_vec, theta_min, theta_max)
    ay_vec = np.clip(ay_vec, theta_min, theta_max)
    return ax_vec, ay_vec


def calibrate_linear_anisotropic(
    params,
    phi_true,
    K,
    theta_min=1.0,
    theta_max=10.0,
    nx_sim=64,
    ny_sim=64,
    num_iters=20,
    lr=0.03,
    grad_eps=1e-2,
):
    """
    Linear functional anisotropic baseline:

        τ_k = k / (K - 1)
        a_x,k = ax0 + sx * τ_k
        a_y,k = ay0 + sy * τ_k

    Optimize (ax0, sx, ay0, sy) via gradient descent with finite-difference
    gradients to minimize concentration MSE at final time.
    """

    def loss_for_params(ax0, sx, ay0, sy):
        ax_vec, ay_vec = build_linear_aniso_profiles(
            ax0, sx, ay0, sy, K, theta_min, theta_max
        )
        return concentration_mse_for_profile(
            params, ax_vec, ay_vec, phi_true, nx_sim, ny_sim
        )

    # Initialize base values at mid-range, zero slopes
    ax0 = 0.5 * (theta_min + theta_max)
    ay0 = 0.5 * (theta_min + theta_max)
    sx = 0.0
    sy = 0.0

    best_params = (ax0, sx, ay0, sy)
    best_loss = loss_for_params(ax0, sx, ay0, sy)

    for it in range(num_iters):
        # Center loss
        loss_center = loss_for_params(ax0, sx, ay0, sy)

        # --- Gradients via central differences ---
        # wrt ax0
        loss_ax0_plus  = loss_for_params(ax0 + grad_eps, sx, ay0, sy)
        loss_ax0_minus = loss_for_params(ax0 - grad_eps, sx, ay0, sy)
        grad_ax0 = (loss_ax0_plus - loss_ax0_minus) / (2.0 * grad_eps)

        # wrt sx
        loss_sx_plus  = loss_for_params(ax0, sx + grad_eps, ay0, sy)
        loss_sx_minus = loss_for_params(ax0, sx - grad_eps, ay0, sy)
        grad_sx = (loss_sx_plus - loss_sx_minus) / (2.0 * grad_eps)

        # wrt ay0
        loss_ay0_plus  = loss_for_params(ax0, sx, ay0 + grad_eps, sy)
        loss_ay0_minus = loss_for_params(ax0, sx, ay0 - grad_eps, sy)
        grad_ay0 = (loss_ay0_plus - loss_ay0_minus) / (2.0 * grad_eps)

        # wrt sy
        loss_sy_plus  = loss_for_params(ax0, sx, ay0, sy + grad_eps)
        loss_sy_minus = loss_for_params(ax0, sx, ay0, sy - grad_eps)
        grad_sy = (loss_sy_plus - loss_sy_minus) / (2.0 * grad_eps)

        # Gradient descent update
        ax0 = ax0 - lr * grad_ax0
        sx  = sx  - lr * grad_sx
        ay0 = ay0 - lr * grad_ay0
        sy  = sy  - lr * grad_sy

        # Clamp base values into [theta_min, theta_max]; slopes remain unconstrained
        ax0 = float(np.clip(ax0, theta_min, theta_max))
        ay0 = float(np.clip(ay0, theta_min, theta_max))

        loss_now = loss_for_params(ax0, sx, ay0, sy)
        if loss_now < best_loss:
            best_loss = loss_now
            best_params = (ax0, sx, ay0, sy)

    return best_params, best_loss


# ============================================================
# 5. Running baselines on the validation split (lin-aniso)
# ============================================================

def run_baselines_on_validation(
    root_dir,
    nx_sim=64,
    ny_sim=64,
    theta_min=1.0,
    theta_max=10.0,
):
    """
    Run both baselines (static and linear anisotropic a_x(t), a_y(t))
    on a validation split of the dataset under root_dir.

    Splitting strategy:
        - All subfolders named 'sim_*' are collected and sorted.
        - First 99.5% are treated as 'train' (ignored here).
        - Remaining 0.5% are treated as 'validation' (to mimic your ML split).
    """
    sim_dirs = sorted(
        d for d in glob.glob(os.path.join(root_dir, "sim_*"))
        if os.path.isdir(d)
    )

    if len(sim_dirs) == 0:
        raise ValueError(f"No sim_* folders found under {root_dir}")

    n_sims = len(sim_dirs)
    n_train = int(0.995 * n_sims)
    val_dirs = sim_dirs[n_train:]

    print(f"Total simulations found: {n_sims}")
    print(f"Using {len(val_dirs)} simulations for validation.\n")

    static_mse_list = []
    linear_mse_list = []

    for sim_dir in val_dirs:
        with open(os.path.join(sim_dir, "params.json"), "r") as f:
            params = json.load(f)

        # Ground-truth stair-step coefficients used in generator
        ax_true_array = np.array(params["coefficients_ax"], dtype=float)
        ay_true_array = np.array(params["coefficients_ay"], dtype=float)
        K = len(ax_true_array)
        assert K == len(ay_true_array)

        # Ground-truth final concentration field
        phi_true = simulate_final_phi_anisotropic(
            params,
            ax_true_array,
            ay_true_array,
            nx=nx_sim,
            ny=ny_sim,
        )

        # ---- Static anisotropic calibration ----
        (ax_const_opt, ay_const_opt), static_mse = calibrate_static_anisotropic(
            params,
            phi_true,
            K,
            theta_min=theta_min,
            theta_max=theta_max,
            nx_sim=nx_sim,
            ny_sim=ny_sim,
        )

        # ---- Linear anisotropic calibration ----
        (ax0_opt, sx_opt, ay0_opt, sy_opt), linear_mse = calibrate_linear_anisotropic(
            params,
            phi_true,
            K,
            theta_min=theta_min,
            theta_max=theta_max,
            nx_sim=nx_sim,
            ny_sim=ny_sim,
        )

        static_mse_list.append(static_mse)
        linear_mse_list.append(linear_mse)

        print(
            f"[{os.path.basename(sim_dir)}] "
            f"Static Conc MSE = {static_mse:.6e}, "
            f"Linear Conc MSE = {linear_mse:.6e} "
            f"(ax_const={ax_const_opt:.4f}, ay_const={ay_const_opt:.4f}, "
            f"ax0={ax0_opt:.4f}, sx={sx_opt:.4f}, ay0={ay0_opt:.4f}, sy={sy_opt:.4f})"
        )

    static_mean = float(np.mean(static_mse_list)) if static_mse_list else float("nan")
    linear_mean = float(np.mean(linear_mse_list)) if linear_mse_list else float("nan")

    print("\n================ Overall Concentration MSE =================")
    print(f"Static anisotropic calibration:  mean Conc MSE = {static_mean:.6e}")
    print(f"Linear anisotropic calibration:  mean Conc MSE = {linear_mean:.6e}")
    print("===========================================================\n")

    return static_mean, linear_mean


# ============================================================
# 6. Main entry point
# ============================================================

def main():
    # TODO: set this to your linear-anisotropic dataset root
    root_dir = "/data/IMcali/sim_results_2025-12-19 04:48:56_lin_ani"  # adjust as needed

    nx_sim = 64
    ny_sim = 64

    # Coefficient range used in your ML setup
    theta_min = 1.0
    theta_max = 10.0

    run_baselines_on_validation(
        root_dir=root_dir,
        nx_sim=nx_sim,
        ny_sim=ny_sim,
        theta_min=theta_min,
        theta_max=theta_max,
    )


if __name__ == "__main__":
    main()
