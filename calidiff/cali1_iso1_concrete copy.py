import os
import json
import glob
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

import numpy as np
from fipy import CellVariable, Grid2D, TransientTerm, DiffusionTerm
from fipy.tools import numerix

DAY_SEC = 24.0 * 3600.0
MM_TO_M = 1e-3

import matplotlib as mpl

# same colormap as in the generator
_base_cmap = mpl.colormaps.get_cmap("nipy_spectral")
_highres_cmap = _base_cmap.resampled(1024)


def phi_to_rgb_colormap(phi_np, vmin=0.0, vmax=0.2):
    """
    Map scalar field φ to RGB using the same colormap and vmin/vmax
    as in the data generator.
    phi_np: [ny, nx] numpy array
    returns: [ny, nx, 3] float32 in [0,1]
    """
    base_cmap = mpl.colormaps.get_cmap("nipy_spectral")
    highres_cmap = base_cmap.resampled(1024)

    phi_clipped = np.clip(phi_np, vmin, vmax)
    norm = (phi_clipped - vmin) / (vmax - vmin + 1e-8)
    rgba = highres_cmap(norm)          # [ny, nx, 4]
    rgb = rgba[..., :3]                # drop alpha
    return rgb.astype(np.float32)


day = 24.0 * 3600.0
MM_TO_M = 1e-3

# Default D_cl in case it's missing in params.json (should match generator)
D_CL_DEFAULT = 5e-12


# -------------------------------------------------------------------------
# Chloride binding function λ(C_f)  (copy of generator logic)
# -------------------------------------------------------------------------
def chloride_binding_lambda(phi, A0_t):
    """
    Compute λ(C_f) for chloride binding:

        λ(C_f) = 10^B * (A0(t) * β_gel / (35450 * β_sol))
                 * ( C_f / (35.45 * β_sol) )^(A0(t) - 1)

    phi: FiPy CellVariable for free chloride C_f
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
# Chloride diffusion simulator (matches generator, but with given A0(t))
# ============================================================
def simulate_final_phi_chloride(params, A0_steps, nx=64, ny=64):
    """
    Re-run the *same* chloride diffusion PDE used in data generation,
    but with a given stair-step A0(t) instead of random sampling.

    params: dict loaded from params.json
      must contain width_mm, height_mm, c_boundary, c_init, c_other, time_days
      and optionally D_cl (else D_CL_DEFAULT is used)
    A0_steps: 1D array-like of length K (N_COEFF_STEPS), scalar A0_k in (0.1, 1.0)
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
    t_final = float(time_days) * day

    t_edges = numerix.linspace(0.0, t_final, num_steps + 1)

    # record initial state (not strictly needed, but matches generator style)
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

            res = 1.0
            #print(A0_t, " ", dt_base, " ", t)
            for _ in range(MAX_SWEEPS):
                res = eqn.sweep(var=phi, dt=dt)
                if res <= TOL:
                    #print("meet TOL")
                    break

            t += dt

    # final φ as numpy array [ny, nx]
    phi_array = numerix.reshape(phi.value, (ny, nx))
    phi_np = np.array(phi_array)
    return phi_np


# ============================================================
# 1. Dataset: Expert Trajectories from sim_* folders (chloride)
# ============================================================

class CalibDiffDataset(Dataset):
    """
    Each item is one (trajectory, time step) pair:
        inputs:  I_k (current), I_goal (final), time index k
        target:  scalar binding exponent A0[k] stored as coefficients_ax[k]
    """

    def __init__(
        self,
        root_dir: str,
        img_size: int = 128,
        K: int = 4,
        theta_min: float = 0.1,
        theta_max: float = 1.0,
    ):
        """
        Args:
            root_dir: folder containing sim_00000_..., sim_00001_..., ...
            img_size: image size after resizing (HxW, assumed square)
            K: number of time intervals (N_COEFF_STEPS in generator)
            theta_min, theta_max: global range for A0(t)
        """
        self.root_dir = root_dir
        self.img_size = img_size
        self.K = K
        self.theta_min = theta_min
        self.theta_max = theta_max

        # Collect all simulation folders
        self.sim_dirs = sorted(
            d for d in glob.glob(os.path.join(root_dir, "sim_*")) if os.path.isdir(d)
        )
        if len(self.sim_dirs) == 0:
            raise ValueError(f"No sim_* folders found under {root_dir}")

        # Basic image transform: resize, grayscale->RGB, to tensor
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),  # [0,1], shape [C,H,W]
        ])

        # Precompute all (sim_idx, k) pairs
        self.index = []  # list of (sim_idx, k)
        for sim_idx in range(len(self.sim_dirs)):
            for k in range(self.K):  # 0..K-1
                self.index.append((sim_idx, k))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        sim_idx, k = self.index[idx]
        sim_dir = self.sim_dirs[sim_idx]

        # --- Load params.json ---
        with open(os.path.join(sim_dir, "params.json"), "r") as f:
            params = json.load(f)

        # --- Load images ---
        # current phi at step k (state at t_k, *before* applying interval k+1)
        phi_k_path = os.path.join(sim_dir, f"phi_interval_{k:02d}.png")
        # goal = final phi at step K (t_{K})
        phi_goal_path = os.path.join(sim_dir, f"phi_interval_{self.K:02d}.png")

        I_k = self.transform(Image.open(phi_k_path).convert("RGB"))        # [3,H,W]
        I_goal = self.transform(Image.open(phi_goal_path).convert("RGB"))  # [3,H,W]

        # --- Target parameter: A0(t) stored as coefficients_ax[k] ---
        A0_k = params["coefficients_ax"][k]
        target = torch.tensor([A0_k], dtype=torch.float32)   # [1]

        # time index (normalized)
        tau_k = torch.tensor([k / (self.K - 1)], dtype=torch.float32)

        return {
            "I_k": I_k,
            "I_goal": I_goal,
            "tau_k": tau_k,
            "target_params": target,  # [1], scalar A0_k
        }


# ============================================================
# 2. CalibDiff model: patch embedding + latent Transformer + 1D classifier over A0
# ============================================================

class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, embed_dim=128, patch_size=8, img_size=128):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.conv = nn.Conv2d(
            in_ch, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: [B,3,H,W] -> [B,C,H_p,W_p]
        x = self.conv(x)
        B, C, H_p, W_p = x.shape
        # flatten to tokens
        x = x.flatten(2).transpose(1, 2)  # [B, N_p, C]
        return x, H_p, W_p


class LatentBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor):
        # x: [B, L, C]
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x_norm = self.norm2(x)
        x = x + self.mlp(x_norm)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.norm_q = nn.LayerNorm(dim)
        self.norm_ctx = nn.LayerNorm(dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor):
        # query: [B, L_q, C], context: [B, L_ctx, C]
        q = self.norm_q(query)
        k = self.norm_ctx(context)
        v = k
        out, _ = self.attn(q, k, v)
        return query + out


class CalibDiff(nn.Module):
    def __init__(
        self,
        img_size=128,
        patch_size=8,
        embed_dim=128,
        num_heads=4,
        num_latents=256,
        num_latent_layers=4,
        theta_min=0.1,
        theta_max=1.0,
        n_bins=8,   # number of classification bins for A0
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.n_bins = n_bins

        # Patch encoders for current and goal (shared)
        self.patch_embed = PatchEmbed(
            in_ch=3, embed_dim=embed_dim, patch_size=patch_size, img_size=img_size
        )

        # Patch grid size
        H_p = img_size // patch_size
        W_p = img_size // patch_size
        self.H_p = H_p
        self.W_p = W_p
        self.N_p = H_p * W_p

        # Modality embeddings: current vs goal
        self.mod_cur = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mod_goal = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Time token (still used as conditioning)
        self.time_token_proj = nn.Linear(1, embed_dim)

        # ---- Unified positional embedding for the full sequence z0 ----
        # Sequence layout: [cur_patches (N_p), goal_patches (N_p), time_token (1)]
        total_tokens = 2 * self.N_p + 1
        self.total_tokens = total_tokens
        self.pos_embed = nn.Parameter(
            torch.randn(total_tokens, embed_dim) * 0.01
        )  # [total_tokens, C]

        # Latent bottleneck
        self.latents = nn.Parameter(torch.randn(1, num_latents, embed_dim) * 0.02)
        self.cross_attn_in = CrossAttention(embed_dim, num_heads)
        self.latent_blocks = nn.ModuleList(
            [LatentBlock(embed_dim, num_heads) for _ in range(num_latent_layers)]
        )
        self.cross_attn_out = CrossAttention(embed_dim, num_heads)

        # ---- Joint head: logits over bins + per-bin offsets ----
        # Output shape: [B, 2 * n_bins] -> split into [logits, offsets]
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 2 * n_bins),
        )

    def forward(self, I_k, I_goal, tau_k):
        """
        Args:
            I_k:    [B,3,H,W]
            I_goal: [B,3,H,W]
            tau_k:  [B,1] normalized time index
        Returns:
            logits:  [B, n_bins]       - classification over bins
            offsets: [B, n_bins]       - local offsets (before scaling)
        """
        B = I_k.shape[0]

        # ---- Patch embedding for current and goal ----
        z_cur, H_p, W_p = self.patch_embed(I_k)     # [B,N_p,C]
        z_goal, _, _ = self.patch_embed(I_goal)     # [B,N_p,C]

        assert H_p == self.H_p and W_p == self.W_p

        # Add modality embeddings
        z_cur = z_cur + self.mod_cur
        z_goal = z_goal + self.mod_goal

        # Concatenate patch tokens
        z_patches = torch.cat([z_cur, z_goal], dim=1)  # [B, 2N_p, C]

        # ---- Time token ----
        z_time = self.time_token_proj(tau_k)  # [B, C]
        z_time = z_time.unsqueeze(1)          # [B,1,C]

        # Full input sequence
        z0 = torch.cat([z_patches, z_time], dim=1)  # [B, N, C]
        N = z0.shape[1]
        assert N == self.total_tokens, f"Expected {self.total_tokens} tokens, got {N}"

        # Positional embeddings
        pos = self.pos_embed.unsqueeze(0).expand(B, -1, -1)  # [B, total_tokens, C]
        z0 = z0 + pos

        # ---- Latent bottleneck ----
        latents = self.latents.expand(B, -1, -1)  # [B,L,C]

        # cross-attn in
        latents = self.cross_attn_in(latents, z0)
        # latent self-attention blocks
        for blk in self.latent_blocks:
            latents = blk(latents)
        # cross-attn out
        zout = self.cross_attn_out(z0, latents)  # [B,N,C]

        # ---- Pool current-image tokens and predict ----
        z_cur_out = zout[:, : self.N_p, :]      # [B,N_p,C]
        z_cur_pool = z_cur_out.mean(dim=1)      # [B,C]

        head_out = self.head(z_cur_pool)        # [B, 2*n_bins]
        logits, offset_raw = head_out.chunk(2, dim=-1)  # [B,n_bins], [B,n_bins]

        # Squash offsets to a small range, e.g. [-0.5, 0.5] in *bin-width units*
        offsets = 0.5 * torch.tanh(offset_raw)  # [B,n_bins]

        return logits, offsets

    # =====================================================
    # 1D discretization for scalar A0
    # =====================================================
    def discretize_1d(self, params: torch.Tensor):
        """
        Convert continuous params [B,1] into 1D bin indices in [0, n_bins - 1],
        using [theta_min, theta_max] mapped onto n_bins bins.
        """
        a = params[:, 0]  # [B]

        v_clamp = torch.clamp(a, self.theta_min, self.theta_max)
        ratio = (v_clamp - self.theta_min) / max(self.theta_max - self.theta_min, 1e-6)

        idx = torch.round(ratio * (self.n_bins - 1)).long()
        idx = torch.clamp(idx, 0, self.n_bins - 1)  # [B]
        return idx

    def bins_to_param(self, bin_idx: torch.Tensor, offsets: torch.Tensor):
        """
        Map bin indices [B] and per-bin offsets [B, n_bins]
        to a continuous scalar a_hat [B].

        offsets are assumed to be in [-0.5, 0.5] (fraction of bin width).
        """
        bin_idx_long = bin_idx.long()
        B = bin_idx_long.shape[0]

        # gather offset corresponding to the selected bin
        offset_sel = offsets.gather(1, bin_idx_long.view(B, 1)).squeeze(1)  # [B]

        # bin centers
        bin_idx_f = bin_idx_long.float()
        bin_width = (self.theta_max - self.theta_min) / self.n_bins
        bin_centers = self.theta_min + (bin_idx_f + 0.5) / self.n_bins * (self.theta_max - self.theta_min)

        # convert offset (fraction of bin width) to actual A0 offset
        # offsets in [-0.5, 0.5] -> +- 0.5 * bin_width
        a_hat = bin_centers + offset_sel * bin_width

        return a_hat

from concurrent.futures import ProcessPoolExecutor
from torch.utils.data import Subset

def _chloride_worker_task(args):
    """
    Runs in a separate CPU process:
      - simulate φ_hat with predicted A0_hat(t)
      - simulate φ_true with ground-truth A0_true(t)
      - compute concentration MSE and RGB MSE
      - compute parameter recovery flag (True/False)
    """
    (
        sim_dir,
        params,
        A0_hat_array,
        K,
        nx_sim,
        ny_sim,
        img_size,
        vmin,
        vmax,
        theta_min,
        theta_max,
    ) = args

    # Ground-truth A0(t)
    A0_true_array = np.array(params["coefficients_ax"], dtype=float)[:K]

    # --- PDE simulations (predicted vs true) ---
    phi_hat = simulate_final_phi_chloride(
        params, A0_hat_array,
        nx=nx_sim, ny=ny_sim,
    )  # [ny_sim, nx_sim]

    phi_true = simulate_final_phi_chloride(
        params, A0_true_array,
        nx=nx_sim, ny=ny_sim,
    )  # [ny_sim, nx_sim]

    # Concentration MSE (scalar field φ)
    conc_mse = float(np.mean((phi_hat - phi_true) ** 2))

    # --- RGB MSE (use same colormap + simple PIL resizing) ---
    # predicted φ -> RGB
    phi_hat_rgb_np = phi_to_rgb_colormap(phi_hat, vmin=vmin, vmax=vmax)  # [ny, nx, 3]
    phi_hat_img = Image.fromarray((phi_hat_rgb_np * 255).astype(np.uint8))
    phi_hat_img = phi_hat_img.resize((img_size, img_size), resample=Image.BILINEAR)
    phi_hat_rgb_resized = np.asarray(phi_hat_img).astype(np.float32) / 255.0  # [H,W,3]

    # goal PNG
    goal_img = Image.open(os.path.join(sim_dir, f"phi_interval_{K:02d}.png")).convert("RGB")
    goal_img = goal_img.resize((img_size, img_size), resample=Image.BILINEAR)
    goal_np = np.asarray(goal_img).astype(np.float32) / 255.0  # [H,W,3]

    rgb_mse = float(np.mean((phi_hat_rgb_resized - goal_np) ** 2))

    # --- parameter recovery flag for this trajectory ---
    theta_range = theta_max - theta_min
    param_tol = 0.1 * theta_range  # 10% of calibration range
    param_success = bool(np.all(np.abs(A0_hat_array - A0_true_array) < param_tol))

    return conc_mse, rgb_mse, param_success

@torch.no_grad()
def eval_epoch_pde(
    model,
    ds_or_subset,
    device: torch.device,
    max_sims: int = 50,
    nx_sim: int = 64,
    ny_sim: int = 64,
    vmin: float = 0.0,
    vmax: float = 0.2,
    num_workers: int = 15,   # <--- 5 CPUs
):
    """
    Physics-based evaluation that compares the *simulated* final
    distribution using predicted A0(t) with the goal.

    Metrics:
        - RGB MSE: MSE between simulated and goal RGB images.
        - Concentration MSE: MSE between simulated and goal concentration
          fields φ(x,y) at final time.
        - Parameter recovery rate: fraction of trajectories where, at every
          time step, |A0_hat_k - A0_true_k| < 10% of calibration range.
    """
    model.eval()

    # --- unwrap Subset if needed ---
    if isinstance(ds_or_subset, Subset):
        base_ds = ds_or_subset.dataset        # CalibDiffDataset
        idx_list = ds_or_subset.indices       # indices into base_ds.index
        sim_idx_set = set()
        for idx in idx_list:
            sim_idx, _ = base_ds.index[idx]   # base_ds.index is (sim_idx, k)
            sim_idx_set.add(sim_idx)
        sim_idx_list = sorted(list(sim_idx_set))
    else:
        base_ds = ds_or_subset
        sim_idx_list = list(range(len(base_ds.sim_dirs)))

    # cap to max_sims
    if max_sims is not None:
        sim_idx_list = sim_idx_list[:max_sims]

    K = base_ds.K
    img_size = base_ds.img_size

    # -----------------------------
    # Stage 1: use model (GPU/CPU) to predict A0_hat(t) for each trajectory
    # -----------------------------
    tasks = []  # arguments for worker processes

    for sim_idx in sim_idx_list:
        sim_dir = base_ds.sim_dirs[sim_idx]

        # --- load params.json ---
        with open(os.path.join(sim_dir, "params.json"), "r") as f:
            params = json.load(f)

        # --- predict A0_hat[k] for k = 0..K-1 ---
        A0_hat_list = []

        for k in range(K):
            phi_k_path    = os.path.join(sim_dir, f"phi_interval_{k:02d}.png")
            phi_goal_path = os.path.join(sim_dir, f"phi_interval_{K:02d}.png")

            # use the same transform as dataset
            I_k    = base_ds.transform(Image.open(phi_k_path).convert("RGB"))
            I_goal = base_ds.transform(Image.open(phi_goal_path).convert("RGB"))
            tau_k  = torch.tensor([k / (K - 1)], dtype=torch.float32)  # [1]

            I_k    = I_k.unsqueeze(0).to(device)       # [1,3,H,W]
            I_goal = I_goal.unsqueeze(0).to(device)    # [1,3,H,W]
            tau_k  = tau_k.unsqueeze(0).to(device)     # [1,1]

            # single forward pass: logits + offsets
            logits, offsets = model(I_k, I_goal, tau_k)       # [1,n_bins], [1,n_bins]
            pred_bins = torch.argmax(logits, dim=-1)          # [1]
            A0_hat = model.bins_to_param(pred_bins, offsets)[0].item()  # scalar

            A0_hat_list.append(A0_hat)

        A0_hat_array = np.array(A0_hat_list, dtype=float)

        # package args for worker
        tasks.append(
            (
                sim_dir,
                params,
                A0_hat_array,
                K,
                nx_sim,
                ny_sim,
                img_size,
                vmin,
                vmax,
                model.theta_min,
                model.theta_max,
            )
        )

    # -----------------------------
    # Stage 2: run PDE + metrics in parallel on CPU workers
    # -----------------------------
    total_rgb_mse = 0.0
    total_conc_mse = 0.0
    successful_recoveries = 0
    n_sims_used = 0

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        for conc_mse, rgb_mse, param_success in ex.map(_chloride_worker_task, tasks):
            total_conc_mse += conc_mse
            total_rgb_mse += rgb_mse
            successful_recoveries += int(param_success)
            n_sims_used += 1

    denom = max(n_sims_used, 1)
    mean_rgb_mse = total_rgb_mse / denom
    mean_conc_mse = total_conc_mse / denom
    param_recovery_rate = successful_recoveries / denom

    return mean_rgb_mse, mean_conc_mse, param_recovery_rate

# ============================================================
# 3. Physics-based evaluation for chloride diffusion
# ============================================================
# @torch.no_grad()
# def eval_epoch_pde(
#     model,
#     ds_or_subset,
#     device: torch.device,
#     max_sims: int = 50,
#     nx_sim: int = 64,
#     ny_sim: int = 64,
#     vmin: float = 0.0,
#     vmax: float = 0.2,
# ):
#     """
#     Physics-based evaluation that compares the *simulated* final
#     distribution using predicted A0(t) with the goal.

#     Metrics:
#         - RGB MSE: MSE between simulated and goal RGB images.
#         - Concentration MSE: MSE between simulated and goal concentration
#           fields φ(x,y) at final time.
#         - Parameter recovery rate: fraction of trajectories where, at every
#           time step, |A0_hat_k - A0_true_k| < 10% of calibration range.
#     """
#     model.eval()

#     # --- unwrap Subset if needed ---
#     if isinstance(ds_or_subset, Subset):
#         base_ds = ds_or_subset.dataset        # CalibDiffDataset
#         idx_list = ds_or_subset.indices       # indices into base_ds.index
#         sim_idx_set = set()
#         for idx in idx_list:
#             sim_idx, _ = base_ds.index[idx]   # base_ds.index is (sim_idx, k)
#             sim_idx_set.add(sim_idx)
#         sim_idx_list = sorted(list(sim_idx_set))
#         sim_idx_list = sim_idx_list[:max_sims]
#     else:
#         base_ds = ds_or_subset
#         sim_idx_list = list(range(min(len(base_ds.sim_dirs), max_sims)))

#     K = base_ds.K
#     img_size = base_ds.img_size

#     total_rgb_mse = 0.0
#     total_conc_mse = 0.0
#     n_sims_used = 0

#     # for parameter recovery rate
#     successful_recoveries = 0
#     theta_range = model.theta_max - model.theta_min
#     param_tol = 0.1 * theta_range  # 10% of calibration range

#     for sim_idx in sim_idx_list:
#         sim_dir = base_ds.sim_dirs[sim_idx]

#         # --- load params.json ---
#         with open(os.path.join(sim_dir, "params.json"), "r") as f:
#             params = json.load(f)

#         # --- predict A0_hat[k] for k = 0..K-1 ---
#         A0_hat_list = []

#         for k in range(K):
#             phi_k_path    = os.path.join(sim_dir, f"phi_interval_{k:02d}.png")
#             phi_goal_path = os.path.join(sim_dir, f"phi_interval_{K:02d}.png")

#             # use the same transform as dataset
#             I_k    = base_ds.transform(Image.open(phi_k_path).convert("RGB"))
#             I_goal = base_ds.transform(Image.open(phi_goal_path).convert("RGB"))
#             tau_k  = torch.tensor([k / (K - 1)], dtype=torch.float32)  # [1]

#             I_k    = I_k.unsqueeze(0).to(device)       # [1,3,H,W]
#             I_goal = I_goal.unsqueeze(0).to(device)    # [1,3,H,W]
#             tau_k  = tau_k.unsqueeze(0).to(device)     # [1,1]

#             # single forward pass: logits + offsets
#             logits, offsets = model(I_k, I_goal, tau_k)       # [1,n_bins], [1,n_bins]
#             pred_bins = torch.argmax(logits, dim=-1)          # [1]
#             A0_hat = model.bins_to_param(pred_bins, offsets)[0].item()  # scalar

#             A0_hat_list.append(A0_hat)

#         A0_hat_array = np.array(A0_hat_list, dtype=float)

#         # ---------- ground-truth A0(t) for this sim ----------
#         A0_true_array = np.array(params["coefficients_ax"], dtype=float)[:K]

#         # ---------- PDE simulation: predicted vs ground truth ----------
#         # predicted A0(t)
#         phi_hat = simulate_final_phi_chloride(
#             params, A0_hat_array,
#             nx=nx_sim, ny=ny_sim,
#         )  # [ny_sim, nx_sim]

#         # ground-truth A0(t)
#         phi_true = simulate_final_phi_chloride(
#             params, A0_true_array,
#             nx=nx_sim, ny=ny_sim,
#         )  # [ny_sim, nx_sim]

#         # Concentration MSE (scalar field φ)
#         conc_mse = float(np.mean((phi_hat - phi_true) ** 2))
#         total_conc_mse += conc_mse

#         # ---------- RGB MSE as before ----------
#         phi_hat_rgb_np = phi_to_rgb_colormap(phi_hat, vmin=vmin, vmax=vmax)  # [ny, nx, 3]
#         phi_hat_rgb = torch.from_numpy(phi_hat_rgb_np).permute(2, 0, 1).unsqueeze(0).float()
#         phi_hat_rgb = F.interpolate(
#             phi_hat_rgb,
#             size=(img_size, img_size),
#             mode="bilinear",
#             align_corners=False,
#         ).to(device)

#         # goal image at final interval (already PNG from generator)
#         goal_img = base_ds.transform(
#             Image.open(os.path.join(sim_dir, f"phi_interval_{K:02d}.png")).convert("RGB")
#         ).unsqueeze(0).to(device)  # [1,3,H,W]

#         # MSE in RGB space
#         rgb_mse = F.mse_loss(phi_hat_rgb, goal_img).item()
#         total_rgb_mse += rgb_mse

#         # ---------- parameter recovery for this trajectory ----------
#         # condition: for all k, |A0_hat_k - A0_true_k| < 0.1 * (theta_max - theta_min)
#         if np.all(np.abs(A0_hat_array - A0_true_array) < param_tol):
#             successful_recoveries += 1

#         n_sims_used += 1

#     denom = max(n_sims_used, 1)
#     mean_rgb_mse = total_rgb_mse / denom
#     mean_conc_mse = total_conc_mse / denom
#     param_recovery_rate = successful_recoveries / denom

#     return mean_rgb_mse, mean_conc_mse, param_recovery_rate


# ============================================================
# 4. Training Loop
# ============================================================
def train_epoch(
    model: CalibDiff,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_every: int = 50,
    curve_path: str = "train_loss_curve2_chloride.png",
):
    model.train()
    total_loss = 0.0
    n_batches = 0

    loss_history = []  # store per-batch loss within this epoch

    for step, batch in enumerate(dataloader, start=1):
        I_k = batch["I_k"].to(device)           # [B,3,H,W]
        I_goal = batch["I_goal"].to(device)
        tau_k = batch["tau_k"].to(device)       # [B,1]
        target_params = batch["target_params"].to(device)  # [B,1], scalar A0

        optimizer.zero_grad()

        logits, offsets = model(I_k, I_goal, tau_k)   # [B,n_bins], [B,n_bins]

        # 1) classification target: bin index of target_params
        target_bins = model.discretize_1d(target_params)      # [B]
        loss_cls = F.cross_entropy(logits, target_bins)

        # 2) regression target: local offset within that bin
        with torch.no_grad():
            # compute bin centers and target local offset fraction in [-0.5,0.5]
            bin_idx_f = target_bins.float()
            bin_width = (model.theta_max - model.theta_min) / model.n_bins
            bin_centers = model.theta_min + (bin_idx_f + 0.5) / model.n_bins * (model.theta_max - model.theta_min)
            offset_target = (target_params.squeeze(1) - bin_centers) / bin_width   # [B]
            offset_target = torch.clamp(offset_target, -0.5, 0.5)

        # predicted offset for the *true* bin
        offset_pred = offsets.gather(1, target_bins.view(-1, 1)).squeeze(1)  # [B]

        loss_reg = F.mse_loss(offset_pred, offset_target)

        # 3) combine
        loss = loss_cls + 0.1 * loss_reg

        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        loss_history.append(loss_value)

        total_loss += loss_value
        n_batches += 1

        # ------------------------------------------------------------------
        # Every `log_every` steps, update and save the training loss curve
        # ------------------------------------------------------------------
        if step % log_every == 0:
            # Downsample: keep only every 10th point for plotting
            stride = 10
            indices = list(range(0, len(loss_history), stride))
            steps_ds = [i + 1 for i in indices]              # 1-based batch indices
            loss_ds = [loss_history[i] for i in indices]

            plt.figure()
            plt.plot(steps_ds, loss_ds, marker="o", linewidth=1)
            plt.xlabel("Batch index (within epoch)")
            plt.ylabel("Training loss")
            plt.title("Training loss curve (current epoch, downsampled)")
            plt.grid(True, linestyle="--", alpha=0.3)

            plt.savefig(curve_path, dpi=150, bbox_inches="tight")
            plt.close()

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_epoch(
    model: CalibDiff,
    dataloader: DataLoader,
    device: torch.device,
):
    model.eval()
    total_loss = 0.0
    total_loss_cls = 0.0
    total_loss_reg = 0.0
    n_batches = 0

    for batch in dataloader:
        I_k = batch["I_k"].to(device)
        I_goal = batch["I_goal"].to(device)
        tau_k = batch["tau_k"].to(device)
        target_params = batch["target_params"].to(device)  # [B,1], continuous A0

        logits, offsets = model(I_k, I_goal, tau_k)   # [B,n_bins], [B,n_bins]

        target_bins = model.discretize_1d(target_params)      # [B]
        loss_cls = F.cross_entropy(logits, target_bins)

        bin_idx_f = target_bins.float()
        bin_width = (model.theta_max - model.theta_min) / model.n_bins
        bin_centers = model.theta_min + (bin_idx_f + 0.5) / model.n_bins * (model.theta_max - model.theta_min)
        offset_target = (target_params.squeeze(1) - bin_centers) / bin_width   # [B]
        offset_target = torch.clamp(offset_target, -0.5, 0.5)

        offset_pred = offsets.gather(1, target_bins.view(-1, 1)).squeeze(1)  # [B]

        loss_reg = F.mse_loss(offset_pred, offset_target)

        loss = loss_cls + 0.1 * loss_reg

        total_loss += loss.item()
        total_loss_cls += loss_cls.item()
        total_loss_reg += loss_reg.item()
        n_batches += 1

    n_batches = max(n_batches, 1)
    return total_loss / n_batches


def plot_all_losses(
    train_losses,
    val_losses,
    pde_losses,
    out_path: str = "loss_curves_all_chloride.png",
):
    """
    Plot epoch-wise train loss, val loss, and PDE RGB MSE in one figure.
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Val loss")
    plt.plot(epochs, pde_losses, label="PDE RGB MSE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / MSE")
    plt.title("Training / Validation / PDE Loss Curves (chloride)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# 5. Main entry point
# ============================================================

def main():
    # TODO: point this to your chloride simulation root (e.g. sim_results_..._concrete_iso)
    root_dir = "/data/IMcali/sim_results_2025-12-19 15:52:24_concrete_iso"
    img_size = 128
    patch_size = 2
    batch_size = 16
    num_epochs = 100

    # A0(t) range in generator: [0.1, 1.0]
    theta_min = 0.1
    theta_max = 1.0
    n_bins = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # Dataset and loaders
    dataset = CalibDiffDataset(
        root_dir=root_dir,
        img_size=img_size,
        theta_min=theta_min,
        theta_max=theta_max,
    )

    # simple split
    n_total = len(dataset)
    n_train = int(0.995 * n_total)
    n_val = n_total - n_train
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # Model with 8 bins for A0
    model = CalibDiff(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=128,
        num_heads=4,
        num_latents=2048,
        num_latent_layers=6,
        theta_min=theta_min,
        theta_max=theta_max,
        n_bins=n_bins,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    train_loss_hist = []
    val_loss_hist = []
    pde_rgb_mse_hist = []
    conc_mse_hist = []
    param_rec_hist = []

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = eval_epoch(model, val_loader, device)
        pde_rgb_mse, conc_mse, param_rec = eval_epoch_pde(model, val_set, device, max_sims=50)

        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        pde_rgb_mse_hist.append(pde_rgb_mse)
        conc_mse_hist.append(conc_mse)
        param_rec_hist.append(param_rec)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss: {train_loss:.4f} | val loss: {val_loss:.4f} | "
            f"PDE RGB MSE: {pde_rgb_mse:.6f} | "
            f"Conc MSE: {conc_mse:.6e} | "
            f"Param recovery: {param_rec*100:.2f}%"
        )

        plot_all_losses(
            train_loss_hist,
            val_loss_hist,
            pde_rgb_mse_hist,
            out_path="loss_curves_all_chloride.png",
        )

    torch.save(model.state_dict(), "calibdiff_chloride_checkpoint.pt")
    print("Training complete, model saved to calibdiff_chloride_checkpoint.pt")
    np.savez(
        "loss_histories_all_chloride.npz",
        train_loss=train_loss_hist,
        val_loss=val_loss_hist,
        pde_rgb_mse=pde_rgb_mse_hist,
        conc_mse=conc_mse_hist,
        param_recovery=param_rec_hist,
    )


if __name__ == "__main__":
    main()
