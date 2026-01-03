import os
import json
import glob

import numpy as np
from fipy import CellVariable, Grid2D, TransientTerm, DiffusionTerm
from fipy.tools import numerix

# ============================================================
# Basic constants (consistent with your generator)
# ============================================================

DAY_SEC = 24.0 * 3600.0
MM_TO_M = 1e-3

# Default D_cl in case it's missing in params.json (should match generator)
D_CL_DEFAULT = 5e-12


# ============================================================
# 1. Chloride binding function λ(C_f)
#    (same logic as in your training script)
# ============================================================

def chloride_binding_lambda(phi, A0_t):
    """
    Compute λ(C_f) for chloride binding:

        λ(C_f) = 10^B * (A0(t) * β_gel / (35450 * β_sol))
                 * ( C_f / (35.45 * β_sol) )^(A0(t) - 1)

    phi : FiPy CellVariable for free chloride C_f
    A0_t: scalar A0(t) in (0.1, 1)
    """
    B = 1.14
    beta_sol = 0.02e-3  # [L/g]
    beta_gel = 0.315    # [g/g]

    # Avoid 0^(A0-1) when A0-1 < 0
    Cf_safe = numerix.maximum(phi, 1e-12)

    base = Cf_safe / (35.45 * beta_sol)
    lam = (10.0 ** B) * (A0_t * beta_gel) / (35450.0 * beta_sol) * base ** (A0_t - 1.0)
    return lam


# ============================================================
# 2. Chloride diffusion simulator (same PDE as generator)
# ============================================================

def simulate_final_phi_chloride(params, A0_steps, nx=64, ny=64):
    """
    Re-run the *same* chloride diffusion PDE used in data generation,
    but with a given stair-step A0(t) instead of random sampling.

    params: dict loaded from params.json
      must contain width_mm, height_mm, c_boundary, c_init, c_other, time_days
      and optionally D_cl (else D_CL_DEFAULT is used)
    A0_steps: 1D array-like of length K (N_COEFF_STEPS), scalar A0_k in (0.1, 1.0)

    Returns:
        phi_np: [ny, nx] numpy array with final free chloride concentration.
    """
    width_mm   = params["width_mm"]
    height_mm  = params["height_mm"]
    c_boundary = params["c_boundary"]
    c_init     = params["c_init"]
    c_other    = params["c_other"]
    time_days  = params["time_days"]
    D_cl       = params.get("D_cl", D_CL_DEFAULT)

    A0_steps = np.asarray(A0_steps, dtype=float)
    num_steps = len(A0_steps)

    # --- Mesh ---
    Lx = width_mm * MM_TO_M
    Ly = height_mm * MM_TO_M

    dx = Lx / nx
    dy = Ly / ny

    mesh = Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)

    # --- Variable: free chloride concentration φ = C_f ---
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

        A0_t = float(A0_steps[k])

        # stability-based dt for explicit diffusion:
        # (1 + λ) * ∂C/∂t = ∇·(D_cl ∇C)
        # worst case λ = 0 => dt <= 0.25 * h^2 / D_cl
        Dmax_ref = D_cl
        Dmax_ref = max(Dmax_ref, 1e-20)
        dt_base  = 0.25 * min(dx, dy)**2 / Dmax_ref

        while t < interval_end - 1e-12:
            dt = min(dt_base, interval_end - t)

            # Binding function λ(C_f) at current A0(t)
            lam_phi = chloride_binding_lambda(phi, A0_t)

            # (1 + λ) * ∂phi/∂t = ∇·(D_cl ∇phi)
            eqn = TransientTerm(coeff=(1.0 + lam_phi)) == DiffusionTerm(coeff=D_cl)

            phi.updateOld()
            TOL = 1e-2
            MAX_SWEEPS = 50
            #print(dt, " ", D_cl, " ", A0_t)
            res = 1.0
            for _ in range(MAX_SWEEPS):
                res = eqn.sweep(var=phi, dt=dt)
                #print(res)
                if res <= TOL:
                    break

            t += dt

    # final φ as numpy array [ny, nx]
    phi_array = numerix.reshape(phi.value, (ny, nx))
    phi_np = np.array(phi_array)
    return phi_np


# ============================================================
# 3. Loss helper: Concentration MSE for a given A0(t)
# ============================================================

def concentration_mse_for_profile(params, A0_steps, phi_true, nx_sim=64, ny_sim=64):
    """
    Given a parameter profile A0_steps and ground-truth final field phi_true,
    simulate the PDE and return the concentration MSE.
    """
    phi_hat = simulate_final_phi_chloride(params, A0_steps, nx=nx_sim, ny=ny_sim)
    mse = float(np.mean((phi_hat - phi_true) ** 2))
    return mse


# ============================================================
# 4. Static parameter calibration (A0(t) ≡ const)
# ============================================================

def calibrate_static_A0(
    params,
    phi_true,
    K,
    theta_min=0.1,
    theta_max=1.0,
    nx_sim=64,
    ny_sim=64,
    num_iters=10,
    lr=0.1,
    grad_eps=1e-3,
):
    """
    Static parameter calibration baseline:
        A0(t) = A0_const  for all time intervals.

    Optimize A0_const via simple gradient descent with finite-difference
    gradient approximation, minimizing concentration MSE at final time.
    """
    def loss_fn(A0_val):
        A0_clamped = float(np.clip(A0_val, theta_min, theta_max))
        A0_vec = np.full(K, A0_clamped, dtype=float)
        return concentration_mse_for_profile(params, A0_vec, phi_true, nx_sim, ny_sim)

    # Initialize at mid-range
    A0 = 0.5 * (theta_min + theta_max)
    best_A0 = A0
    best_loss = loss_fn(A0)

    for it in range(num_iters):
        # Finite-difference gradient
        loss_plus = loss_fn(A0 + grad_eps)
        loss_minus = loss_fn(A0 - grad_eps)
        grad = (loss_plus - loss_minus) / (2.0 * grad_eps)

        # Gradient descent update
        A0 = A0 - lr * grad
        A0 = float(np.clip(A0, theta_min, theta_max))

        loss_now = loss_fn(A0)
        if loss_now < best_loss:
            best_loss = loss_now
            best_A0 = A0

    return best_A0, best_loss


# ============================================================
# 5. Linear functional parameter calibration
#    A0(t) = A0_0 + slope * τ,   τ ∈ [0,1] normalized time
# ============================================================

def build_linear_A0_profile(A0_0, slope, K, theta_min=0.1, theta_max=1.0):
    """
    Construct a linear-in-time profile for A0(t) over K intervals:
        τ_k = k / (K - 1),  k = 0,...,K-1
        A0_k = A0_0 + slope * τ_k
    Then clamp A0_k to [theta_min, theta_max].
    """
    if K == 1:
        t_norm = np.array([0.0], dtype=float)
    else:
        t_norm = np.linspace(0.0, 1.0, K, dtype=float)

    A0_vec = A0_0 + slope * t_norm
    A0_vec = np.clip(A0_vec, theta_min, theta_max)
    return A0_vec


def calibrate_linear_A0(
    params,
    phi_true,
    K,
    theta_min=0.1,
    theta_max=1.0,
    nx_sim=64,
    ny_sim=64,
    num_iters=20,
    lr=0.05,
    grad_eps=1e-3,
):
    """
    Prescribed linear functional calibration baseline:

        A0_k = A0_0 + slope * τ_k,    τ_k = k / (K-1)

    Optimize (A0_0, slope) via gradient descent with finite-difference
    gradients to minimize concentration MSE at final time.
    """
    def loss_for_params(A0_0, slope):
        A0_vec = build_linear_A0_profile(A0_0, slope, K, theta_min, theta_max)
        return concentration_mse_for_profile(params, A0_vec, phi_true, nx_sim, ny_sim)

    # Initialize at mid-range with zero slope (same as static mid-range guess)
    A0_0 = 0.5 * (theta_min + theta_max)
    slope = 0.0

    best_params = (A0_0, slope)
    best_loss = loss_for_params(A0_0, slope)

    for it in range(num_iters):
        # Finite-difference gradients
        loss_center = loss_for_params(A0_0, slope)

        # Gradient wrt A0_0
        loss_A_plus = loss_for_params(A0_0 + grad_eps, slope)
        loss_A_minus = loss_for_params(A0_0 - grad_eps, slope)
        grad_A0 = (loss_A_plus - loss_A_minus) / (2.0 * grad_eps)

        # Gradient wrt slope
        loss_s_plus = loss_for_params(A0_0, slope + grad_eps)
        loss_s_minus = loss_for_params(A0_0, slope - grad_eps)
        grad_s = (loss_s_plus - loss_s_minus) / (2.0 * grad_eps)

        # Gradient descent update
        A0_0 = A0_0 - lr * grad_A0
        slope = slope - lr * grad_s

        # Optionally clamp A0_0 to range; slope left unconstrained
        A0_0 = float(np.clip(A0_0, theta_min, theta_max))

        loss_now = loss_for_params(A0_0, slope)
        if loss_now < best_loss:
            best_loss = loss_now
            best_params = (A0_0, slope)

    return best_params, best_loss


# ============================================================
# 6. Running baselines on the "validation" dataset
# ============================================================

def run_baselines_on_validation(
    root_dir,
    nx_sim=64,
    ny_sim=64,
    theta_min=0.1,
    theta_max=1.0,
):
    """
    Run both baselines (static and linear A0(t)) on a validation split
    of the dataset under root_dir.

    Splitting strategy:
        - All subfolders named 'sim_*' are collected and sorted.
        - First 90% are treated as 'train' (ignored here).
        - Remaining 10% are treated as 'validation'.

    For each validation simulation:
        - Ground-truth A0(t) = params['coefficients_ax'].
        - Compute φ_true at final time with simulate_final_phi_chloride.
        - Calibrate A0_const and (A0_0, slope) baselines.
        - Print per-simulation Concentration MSE.
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
    print(f"Using {n_sims - n_train} simulations for validation.\n")

    static_mse_list = []
    linear_mse_list = []

    for sim_dir in val_dirs:
        with open(os.path.join(sim_dir, "params.json"), "r") as f:
            params = json.load(f)

        # Ground-truth A0(t) used in generator
        A0_true_array = np.array(params["coefficients_ax"], dtype=float)
        K = len(A0_true_array)

        # Ground-truth final concentration field
        phi_true = simulate_final_phi_chloride(
            params,
            A0_true_array,
            nx=nx_sim,
            ny=ny_sim,
        )

        # ---- Static calibration ----
        A0_const_opt, static_mse = calibrate_static_A0(
            params,
            phi_true,
            K,
            theta_min=theta_min,
            theta_max=theta_max,
            nx_sim=nx_sim,
            ny_sim=ny_sim,
        )

        # ---- Linear functional calibration ----
        (A0_0_opt, slope_opt), linear_mse = calibrate_linear_A0(
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
            f"Static MSE = {static_mse:.6e}, "
            f"Linear MSE = {linear_mse:.6e} "
            f"(A0_const={A0_const_opt:.4f}, A0_0={A0_0_opt:.4f}, slope={slope_opt:.4f})"
        )

    static_mean = float(np.mean(static_mse_list)) if static_mse_list else float("nan")
    linear_mean = float(np.mean(linear_mse_list)) if linear_mse_list else float("nan")

    print("\n================ Overall Concentration MSE =================")
    print(f"Static parameter calibration:   mean Conc MSE = {static_mean:.6e}")
    print(f"Linear functional calibration:  mean Conc MSE = {linear_mean:.6e}")
    print("===========================================================\n")

    return static_mean, linear_mean


# ============================================================
# 7. Main
# ============================================================

def main():
    # TODO: set this to your chloride dataset root
    root_dir = "/data/IMcali/sim_results_2025-12-19 15:52:24_concrete_iso"

    nx_sim = 64
    ny_sim = 64

    # A0(t) range in the synthetic generator
    theta_min = 0.1
    theta_max = 1.0

    run_baselines_on_validation(
        root_dir=root_dir,
        nx_sim=nx_sim,
        ny_sim=ny_sim,
        theta_min=theta_min,
        theta_max=theta_max,
    )


if __name__ == "__main__":
    main()
