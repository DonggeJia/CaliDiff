import os
import json
import glob
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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

import numpy as np
import matplotlib as mpl
from fipy import CellVariable, Grid2D, TransientTerm, DiffusionTerm
from fipy.tools import numerix

# Make sure these match your generator:
D28x = 1e-12       # same as in data generation
N_POLY = 2.0
day   = 24.0 * 3600.0
MM_TO_M = 1e-3


def simulate_final_phi_isotropic(params, a_steps, nx=64, ny=64):
    """
    Re-run the *same* isotropic diffusion PDE you used for data generation,
    but with a *given* stair-step a(t) instead of random sampling.

    params: dict loaded from params.json
      must contain width_mm, height_mm, c_boundary, c_init, c_other, time_days
    a_steps: 1D array-like of length K (N_COEFF_STEPS), isotropic scalar a_k
    """
    width_mm   = params["width_mm"]
    height_mm  = params["height_mm"]
    c_boundary = params["c_boundary"]
    c_init     = params["c_init"]
    c_other    = params["c_other"]
    time_days  = params["time_days"]

    a_steps = np.asarray(a_steps, dtype=float)
    num_steps = len(a_steps)

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
    t_final = float(time_days) * day

    # Use len(a_steps) as N_COEFF_STEPS to avoid hard-coding
    t_edges = numerix.linspace(0.0, t_final, num_steps + 1)

    for k in range(num_steps):
        interval_start = float(t_edges[k])
        interval_end   = float(t_edges[k + 1])

        a_t = float(a_steps[k])

        # stability-based dt (same as your generator logic)
        Dmax_ref = D28x * (1.0 + abs(a_t))
        Dmax_ref = max(Dmax_ref, 1e-20)
        dt_base  = 0.25 * min(dx, dy)**2 / Dmax_ref

        while t < interval_end - 1e-12:
            dt = min(dt_base, interval_end - t)

            # Isotropic, nonlinear D(φ)
            Dx_phi = D28x * (1.0 + 5000 * a_t * phi**N_POLY)
            Dy_phi = Dx_phi  # isotropic

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
# 1. Dataset: Expert Trajectories from sim_* folders
# ============================================================

class CalibDiffDataset(Dataset):
    """
    Each item is one (trajectory, time step) pair:
        inputs:  I_k (current), I_goal (final), time index k
        target:  scalar diffusion parameter a[k]
    """

    def __init__(
        self,
        root_dir: str,
        img_size: int = 128,
        K: int = 4,
        theta_min: float = 0.1,
        theta_max: float = 10.0,
    ):
        """
        Args:
            root_dir: folder containing sim_00000_..., sim_00001_..., ...
            img_size: image size after resizing (HxW, assumed square)
            K: number of time intervals (3 in your simulation)
            theta_min, theta_max: global range for coefficient a
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

        # --- Target parameter (isotropic) ---
        # Generator saves coefficients_ax and coefficients_ay but they are equal.
        a = params["coefficients_ax"][k]
        target = torch.tensor([a], dtype=torch.float32)   # [1]

        # time index (normalized)
        tau_k = torch.tensor([k / (self.K - 1)], dtype=torch.float32)

        return {
            "I_k": I_k,
            "I_goal": I_goal,
            "tau_k": tau_k,
            "target_params": target,  # [1], scalar a
        }


# ============================================================
# 2. CalibDiff model: patch embedding + latent Transformer + 1D classifier over a
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
        theta_max=10.0,
        n_bins=8,   # <--- number of classification bins for a
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

        # ---- Classification head: embed_dim -> N_BINS ----
        self.cls_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, n_bins),
        )

    def forward(self, I_k, I_goal, tau_k):
        """
        Args:
            I_k:    [B,3,H,W]
            I_goal: [B,3,H,W]
            tau_k:  [B,1] normalized time index
        Returns:
            logits: [B, N_BINS] over the scalar parameter a
        """
        B = I_k.shape[0]

        # ---- Patch embedding for current and goal ----
        z_cur, H_p, W_p = self.patch_embed(I_k)     # [B,N_p,C]
        z_goal, _, _ = self.patch_embed(I_goal)     # [B,N_p,C]

        # sanity check
        assert H_p == self.H_p and W_p == self.W_p

        # Add modality embeddings to patches (no positions yet)
        z_cur = z_cur + self.mod_cur               # [B,N_p,C]
        z_goal = z_goal + self.mod_goal            # [B,N_p,C]

        # Concatenate patch tokens
        z_patches = torch.cat([z_cur, z_goal], dim=1)  # [B, 2N_p, C]

        # ---- Time token ----
        z_time = self.time_token_proj(tau_k)  # [B, C]
        z_time = z_time.unsqueeze(1)          # [B,1,C]

        # Full input sequence: patches + time token
        z0 = torch.cat([z_patches, z_time], dim=1)  # [B, N, C]
        N = z0.shape[1]
        assert N == self.total_tokens, f"Expected {self.total_tokens} tokens, got {N}"

        # ---- Add unified positional embeddings to the entire sequence ----
        pos = self.pos_embed.unsqueeze(0).expand(B, -1, -1)  # [B, total_tokens, C]
        z0 = z0 + pos

        # ---- Latent bottleneck ----
        latents = self.latents.expand(B, -1, -1)  # [B,L,C]

        # cross-attn in (latents query, tokens context)
        latents = self.cross_attn_in(latents, z0)
        # latent self-attention blocks
        for blk in self.latent_blocks:
            latents = blk(latents)
        # cross-attn out (tokens query, latents context)
        zout = self.cross_attn_out(z0, latents)  # [B,N,C]

        # ---- Pool current-image tokens and classify ----
        # current tokens are first N_p entries
        z_cur_out = zout[:, : self.N_p, :]  # [B,N_p,C]
        z_cur_pool = z_cur_out.mean(dim=1)  # [B,C] - simple average pooling over patches

        logits = self.cls_head(z_cur_pool)  # [B, n_bins]
        return logits

    # =====================================================
    # 1D discretization for scalar a
    # =====================================================
    def discretize_1d(self, params: torch.Tensor):
        """
        Convert continuous params [B,1] into 1D bin indices in [0, n_bins - 1],
        using [theta_min, theta_max] mapped onto n_bins bins.
        """
        a = params[:, 0]  # [B]

        v_clamp = torch.clamp(a, self.theta_min, self.theta_max)
        ratio = (v_clamp - self.theta_min) / max(self.theta_max - self.theta_min, 1e-6)

        n_bins = self.n_bins
        idx = torch.round(ratio * (n_bins - 1)).long()
        idx = torch.clamp(idx, 0, n_bins - 1)  # [B]
        return idx

    def bins_to_param(self, idx: torch.Tensor):
        """
        Map 1D bin indices [B] back to scalar parameter a_hat (bin centers).
        """
        idx = idx.float()
        n_bins = self.n_bins
        a_hat = self.theta_min + (idx + 0.5) / n_bins * (self.theta_max - self.theta_min)
        return a_hat

import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from PIL import Image
import numpy as np


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
):
    """
    Physics-based evaluation that compares the *simulated* final
    distribution using predicted a(t) with the goal PNG.

    ds_or_subset:
        - CalibDiffDataset
        - or torch.utils.data.Subset wrapping CalibDiffDataset

    Returns:
        mean_image_loss (float): MSE in RGB space between simulated final
        distribution and goal image (downsampled/upsampled to img_size).
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
        sim_idx_list = sim_idx_list[:max_sims]
    else:
        base_ds = ds_or_subset
        sim_idx_list = list(range(min(len(base_ds.sim_dirs), max_sims)))

    K = base_ds.K
    img_size = base_ds.img_size

    total_loss = 0.0
    n_sims_used = 0

    for sim_idx in sim_idx_list:
        sim_dir = base_ds.sim_dirs[sim_idx]

        # --- load params.json ---
        with open(os.path.join(sim_dir, "params.json"), "r") as f:
            params = json.load(f)

        # --- predict a_hat[k] for k = 0..K-1 ---
        a_hat_list = []

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

            logits = model(I_k, I_goal, tau_k)         # [1, n_bins]
            pred_bins = torch.argmax(logits, dim=-1)   # [1]

            # scalar a_hat at the bin center
            a_hat = model.bins_to_param(pred_bins)[0].item()
            a_hat_list.append(a_hat)

        a_hat_array = np.array(a_hat_list, dtype=float)

        # --- FiPy simulate with these predicted a's ---
        phi_hat = simulate_final_phi_isotropic(
            params, a_hat_array,
            nx=nx_sim, ny=ny_sim,
        )  # [ny_sim, nx_sim]

        # map to RGB with same colormap
        phi_hat_rgb_np = phi_to_rgb_colormap(phi_hat, vmin=vmin, vmax=vmax)  # [ny, nx, 3]

        # to tensor [1,3,H_sim,W_sim]
        phi_hat_rgb = torch.from_numpy(phi_hat_rgb_np).permute(2, 0, 1).unsqueeze(0).float()
        phi_hat_rgb = F.interpolate(
            phi_hat_rgb,
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        ).to(device)

        # goal image at final interval (already PNG from generator)
        goal_img = base_ds.transform(
            Image.open(os.path.join(sim_dir, f"phi_interval_{K:02d}.png")).convert("RGB")
        ).unsqueeze(0).to(device)  # [1,3,H,W]

        # MSE in RGB space
        loss_img = F.mse_loss(phi_hat_rgb, goal_img)
        total_loss += loss_img.item()
        n_sims_used += 1

    return total_loss / max(n_sims_used, 1)

# ============================================================
# 3. Training Loop
# ============================================================
def train_epoch(
    model: CalibDiff,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_every: int = 50,
    curve_path: str = "train_loss_curve2.png",
):
    model.train()
    total_loss = 0.0
    n_batches = 0

    loss_history = []  # store per-batch loss within this epoch

    for step, batch in enumerate(dataloader, start=1):
        I_k = batch["I_k"].to(device)           # [B,3,H,W]
        I_goal = batch["I_goal"].to(device)
        tau_k = batch["tau_k"].to(device)       # [B,1]
        target_params = batch["target_params"].to(device)  # [B,1], scalar a

        optimizer.zero_grad()

        # forward
        logits = model(I_k, I_goal, tau_k)  # [B, n_bins]

        # discretize ground-truth scalar a to 1D bin index
        target_flat = model.discretize_1d(target_params)  # [B]

        loss = F.cross_entropy(logits, target_flat)
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
    n_batches = 0

    for batch in dataloader:
        I_k = batch["I_k"].to(device)
        I_goal = batch["I_goal"].to(device)
        tau_k = batch["tau_k"].to(device)
        target_params = batch["target_params"].to(device)  # [B,1]

        logits = model(I_k, I_goal, tau_k)  # [B, n_bins]
        target_flat = model.discretize_1d(target_params)

        loss = F.cross_entropy(logits, target_flat)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def plot_all_losses(
    train_losses,
    val_losses,
    pde_losses,
    out_path: str = "loss_curves_all.png",
):
    """
    Plot epoch-wise train CE, val CE, and PDE RGB MSE in one figure.
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_losses, label="Train CE")
    plt.plot(epochs, val_losses, label="Val CE")
    plt.plot(epochs, pde_losses, label="PDE RGB MSE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation / PDE Loss Curves")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

# ============================================================
# 4. Main entry point
# ============================================================

def main():
    root_dir = "/data/IMcali/sim_results_2025-12-16 04:50:13"  # TODO: set your dataset path
    img_size = 128
    patch_size = 2
    batch_size = 16
    num_epochs = 100
    theta_min = 0.1    # tune based on your coefficient ranges
    theta_max = 10.0
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
    n_train = int(0.9 * n_total)
    n_val = n_total - n_train
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # Model with 8 bins
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
    pde_loss_hist = []

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_ce_loss = eval_epoch(model, val_loader, device)
        pde_img_loss = eval_epoch_pde(model, val_set, device, max_sims=50)

        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_ce_loss)
        pde_loss_hist.append(pde_img_loss)

        print(
            f"Epoch {epoch:03d} | "
            f"train CE: {train_loss:.4f} | val CE: {val_ce_loss:.4f} | "
            f"PDE RGB MSE: {pde_img_loss:.6f}"
        )

        plot_all_losses(
            train_loss_hist,
            val_loss_hist,
            pde_loss_hist,
            out_path="loss_curves_all.png",
        )

    torch.save(model.state_dict(), "calibdiff_polynomial_checkpoint.pt")
    print("Training complete, model saved to calibdiff_polynomial_checkpoint.pt")


if __name__ == "__main__":
    main()
