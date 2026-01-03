import os
import json
import glob
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

import matplotlib.pyplot as plt
import matplotlib as mpl

from concurrent.futures import ProcessPoolExecutor

# -------------------------------------------------------------------------
# External diffusion-family modules (single-family anisotropic scripts)
# -------------------------------------------------------------------------
import cali1_iso1_lin_ani as lin_ani_mod
import cali1_iso1_exp_ani as exp_ani_mod


DAY_SEC = 24.0 * 3600.0

# same colormap as in the generators
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


# -------------------------------------------------------------------------
# 1. Multi-family dataset (2D anisotropic parameters)
# -------------------------------------------------------------------------

class MultiFamilyCalibDiffDataset(Dataset):
    """
    Mixed dataset over multiple anisotropic diffusion 'families', each with
    2D parameters per time interval: a_x(t), a_y(t), stored in
        params["coefficients_ax"], params["coefficients_ay"].

    Each item is one (trajectory, time step) pair:
        inputs:  I_k (current), I_goal (final), normalized time index tau_k
        target:  [a_x, a_y] at that interval
        meta:    family name, sim_idx, k
    """
    def __init__(
        self,
        family_roots: Dict[str, str],
        img_size: int = 128,
    ):
        """
        Args:
            family_roots: mapping from family name to sim_results root dir, e.g.
                {
                    "lin_ani": "/data/..._lin_ani",
                    "exp_ani": "/data/..._exp_ani",
                }
            img_size: resize all PNGs to (img_size, img_size)
        """
        super().__init__()
        self.family_roots = dict(family_roots)
        self.img_size = img_size

        # basic transform: resize + ToTensor
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),          # [0,1], [C,H,W]
        ])

        # For each family: collect sim dirs, read K from first params.json,
        # and build a global index list.
        self.family_info: Dict[str, Dict] = {}
        self.items: List[Tuple[str, int, int]] = []  # (family, sim_idx, k)

        for fam, root in self.family_roots.items():
            sim_dirs = sorted(
                d for d in glob.glob(os.path.join(root, "sim_*"))
                if os.path.isdir(d)
            )
            if len(sim_dirs) == 0:
                raise ValueError(f"No sim_* folders found under {root} for family '{fam}'")

            # assume K is constant within a family; read from first sim
            first_params_path = os.path.join(sim_dirs[0], "params.json")
            with open(first_params_path, "r") as f:
                params0 = json.load(f)
            K = len(params0["coefficients_ax"])
            assert len(params0["coefficients_ay"]) == K, \
                "coefficients_ax and coefficients_ay must have same length"

            self.family_info[fam] = {
                "root_dir": root,
                "sim_dirs": sim_dirs,
                "K": K,
            }

            for sim_idx in range(len(sim_dirs)):
                for k in range(K):
                    self.items.append((fam, sim_idx, k))

        # mirrors per-family dataset "index" concept
        self.index = self.items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fam, sim_idx, k = self.items[idx]
        fam_info = self.family_info[fam]
        sim_dir = fam_info["sim_dirs"][sim_idx]
        K = fam_info["K"]

        # params.json holds coefficients_ax and coefficients_ay
        with open(os.path.join(sim_dir, "params.json"), "r") as f:
            params = json.load(f)

        a_x_k = params["coefficients_ax"][k]
        a_y_k = params["coefficients_ay"][k]
        target = torch.tensor([a_x_k, a_y_k], dtype=torch.float32)   # [2]

        # images
        phi_k_path = os.path.join(sim_dir, f"phi_interval_{k:02d}.png")
        phi_goal_path = os.path.join(sim_dir, f"phi_interval_{K:02d}.png")

        I_k = self.transform(Image.open(phi_k_path).convert("RGB"))       # [3,H,W]
        I_goal = self.transform(Image.open(phi_goal_path).convert("RGB")) # [3,H,W]

        tau_k = torch.tensor([k / (K - 1)], dtype=torch.float32)

        return {
            "I_k": I_k,
            "I_goal": I_goal,
            "tau_k": tau_k,
            "target_params": target,   # [2] = [a_x, a_y]
            "family": fam,
            "sim_idx": sim_idx,
            "k": k,
        }


# -------------------------------------------------------------------------
# 2. CalibDiff model (ablated: NO language conditioning; anisotropic 2D param)
# -------------------------------------------------------------------------

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
        theta_min=1.0,      # global range across anisotropic families
        theta_max=10.0,
        n_bins=8,           # number of bins per dimension
    ):
        """
        CalibDiff WITHOUT language conditioning, for 2D anisotropic params.

        Theta range is shared across all families: [1.0, 10.0].
        """
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

        # Time token
        self.time_token_proj = nn.Linear(1, embed_dim)

        # Unified positional embedding:
        #   [cur_patches (N_p), goal_patches (N_p), time_token (1)]
        total_tokens = 2 * self.N_p + 1
        self.total_tokens = total_tokens
        self.pos_embed = nn.Parameter(
            torch.randn(total_tokens, embed_dim) * 0.01
        )

        # Latent bottleneck
        self.latents = nn.Parameter(torch.randn(1, num_latents, embed_dim) * 0.02)
        self.cross_attn_in = CrossAttention(embed_dim, num_heads)
        self.latent_blocks = nn.ModuleList(
            [LatentBlock(embed_dim, num_heads) for _ in range(num_latent_layers)]
        )
        self.cross_attn_out = CrossAttention(embed_dim, num_heads)

        # Head: logits / offsets for x and y, each with n_bins bins
        # Outputs: [B, 4 * n_bins] -> split into
        #   logits_x, offsets_x, logits_y, offsets_y
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 4 * n_bins),
        )

    def forward(self, I_k, I_goal, tau_k):
        """
        Args:
            I_k:    [B,3,H,W]
            I_goal: [B,3,H,W]
            tau_k:  [B,1] normalized time index
        Returns:
            logits_x:   [B, n_bins]
            offsets_x:  [B, n_bins]
            logits_y:   [B, n_bins]
            offsets_y:  [B, n_bins]
        """
        B = I_k.shape[0]

        # patch embeddings
        z_cur, H_p, W_p = self.patch_embed(I_k)   # [B,N_p,C]
        z_goal, _, _ = self.patch_embed(I_goal)   # [B,N_p,C]

        assert H_p == self.H_p and W_p == self.W_p

        z_cur = z_cur + self.mod_cur
        z_goal = z_goal + self.mod_goal

        z_patches = torch.cat([z_cur, z_goal], dim=1)  # [B,2N_p,C]

        # time token
        z_time = self.time_token_proj(tau_k)   # [B,C]
        z_time = z_time.unsqueeze(1)           # [B,1,C]

        z0 = torch.cat([z_patches, z_time], dim=1)  # [B,2N_p+1,C]

        N = z0.shape[1]
        assert N == self.total_tokens, f"Expected {self.total_tokens} tokens, got {N}"

        # positional embeddings
        pos = self.pos_embed.unsqueeze(0).expand(B, -1, -1)
        z0 = z0 + pos

        # latent bottleneck
        latents = self.latents.expand(B, -1, -1)
        latents = self.cross_attn_in(latents, z0)
        for blk in self.latent_blocks:
            latents = blk(latents)
        zout = self.cross_attn_out(z0, latents)  # [B,N,C]

        # pool current-image tokens
        z_cur_out = zout[:, : self.N_p, :]
        z_cur_pool = z_cur_out.mean(dim=1)       # [B,C]

        head_out = self.head(z_cur_pool)         # [B,4*n_bins]
        logits_x, offset_x_raw, logits_y, offset_y_raw = torch.chunk(
            head_out, 4, dim=-1
        )

        offsets_x = 0.5 * torch.tanh(offset_x_raw)  # [-0.5,0.5]
        offsets_y = 0.5 * torch.tanh(offset_y_raw)  # [-0.5,0.5]

        return logits_x, offsets_x, logits_y, offsets_y

    # ---------------- 2D discretization helpers -----------------

    def discretize_2d(self, params: torch.Tensor):
        """
        Convert continuous params [B,2] into 1D bin indices for x and y:
            idx_x, idx_y in [0..n_bins-1]
        using the global [theta_min, theta_max].
        """
        a_x = params[:, 0]  # [B]
        a_y = params[:, 1]  # [B]

        v_clamp_x = torch.clamp(a_x, self.theta_min, self.theta_max)
        v_clamp_y = torch.clamp(a_y, self.theta_min, self.theta_max)

        ratio_x = (v_clamp_x - self.theta_min) / max(self.theta_max - self.theta_min, 1e-6)
        ratio_y = (v_clamp_y - self.theta_min) / max(self.theta_max - self.theta_min, 1e-6)

        idx_x = torch.round(ratio_x * (self.n_bins - 1)).long()
        idx_y = torch.round(ratio_y * (self.n_bins - 1)).long()

        idx_x = torch.clamp(idx_x, 0, self.n_bins - 1)
        idx_y = torch.clamp(idx_y, 0, self.n_bins - 1)

        return idx_x, idx_y

    def bins_to_param_2d(
        self,
        bin_idx_x: torch.Tensor,
        offsets_x: torch.Tensor,
        bin_idx_y: torch.Tensor,
        offsets_y: torch.Tensor,
    ):
        """
        Given per-dimension bin indices [B] and offsets [B,n_bins],
        map to continuous scalar parameters a_x_hat, a_y_hat, shape [B,2].
        """
        B = bin_idx_x.shape[0]

        bin_idx_x_long = bin_idx_x.long()
        bin_idx_y_long = bin_idx_y.long()

        # gather offsets
        offset_x_sel = offsets_x.gather(1, bin_idx_x_long.view(B, 1)).squeeze(1)  # [B]
        offset_y_sel = offsets_y.gather(1, bin_idx_y_long.view(B, 1)).squeeze(1)  # [B]

        bin_width = (self.theta_max - self.theta_min) / self.n_bins

        bin_centers_x = self.theta_min + (
            (bin_idx_x_long.float() + 0.5) / self.n_bins
        ) * (self.theta_max - self.theta_min)

        bin_centers_y = self.theta_min + (
            (bin_idx_y_long.float() + 0.5) / self.n_bins
        ) * (self.theta_max - self.theta_min)

        a_x_hat = bin_centers_x + offset_x_sel * bin_width
        a_y_hat = bin_centers_y + offset_y_sel * bin_width

        a_hat = torch.stack([a_x_hat, a_y_hat], dim=-1)  # [B,2]
        return a_hat


# -------------------------------------------------------------------------
# 3. Multi-family anisotropic PDE worker & evaluation (NO language)
# -------------------------------------------------------------------------

def _multi_aniso_worker_task(args):
    """
    Runs in a separate CPU process:
      - simulate φ_hat with predicted a_hat_x(t), a_hat_y(t)
      - simulate φ_true with ground-truth a_true_x(t), a_true_y(t)
      - compute MSEs and parameter recovery flag
    """
    (
        family,
        sim_dir,
        params,
        ax_hat_array,
        ay_hat_array,
        K,
        nx_sim,
        ny_sim,
        img_size,
        vmin,
        vmax,
        theta_min,
        theta_max,
    ) = args

    # ground truth
    ax_true_array = np.array(params["coefficients_ax"], dtype=float)[:K]
    ay_true_array = np.array(params["coefficients_ay"], dtype=float)[:K]

    # choose PDE simulator
    if family == "lin_ani":
        phi_hat = lin_ani_mod.simulate_final_phi_anisotropic(
            params, ax_hat_array, ay_hat_array, nx=nx_sim, ny=ny_sim
        )
        phi_true = lin_ani_mod.simulate_final_phi_anisotropic(
            params, ax_true_array, ay_true_array, nx=nx_sim, ny=ny_sim
        )
    elif family == "exp_ani":
        phi_hat = exp_ani_mod.simulate_final_phi_anisotropic(
            params, ax_hat_array, ay_hat_array, nx=nx_sim, ny=ny_sim
        )
        phi_true = exp_ani_mod.simulate_final_phi_anisotropic(
            params, ax_true_array, ay_true_array, nx=nx_sim, ny=ny_sim
        )
    else:
        raise ValueError(f"Unknown family {family}")

    conc_mse = float(np.mean((phi_hat - phi_true) ** 2))

    # RGB MSE
    phi_hat_rgb_np = phi_to_rgb_colormap(phi_hat, vmin=vmin, vmax=vmax)
    phi_hat_img = Image.fromarray((phi_hat_rgb_np * 255).astype(np.uint8))
    phi_hat_img = phi_hat_img.resize((img_size, img_size), resample=Image.BILINEAR)
    phi_hat_rgb_resized = np.asarray(phi_hat_img).astype(np.float32) / 255.0

    goal_img = Image.open(os.path.join(sim_dir, f"phi_interval_{K:02d}.png")).convert("RGB")
    goal_img = goal_img.resize((img_size, img_size), resample=Image.BILINEAR)
    goal_np = np.asarray(goal_img).astype(np.float32) / 255.0

    rgb_mse = float(np.mean((phi_hat_rgb_resized - goal_np) ** 2))

    # parameter recovery: 10% of global calibration range (1..10)
    theta_range = theta_max - theta_min
    param_tol = 0.1 * theta_range

    ok_x = np.all(np.abs(ax_hat_array - ax_true_array) < param_tol)
    ok_y = np.all(np.abs(ay_hat_array - ay_true_array) < param_tol)
    param_success = bool(ok_x and ok_y)

    return conc_mse, rgb_mse, param_success


@torch.no_grad()
def eval_epoch_pde_multi(
    model: CalibDiff,
    ds_or_subset,
    device: torch.device,
    max_sims: int = 50,
    nx_sim: int = 64,
    ny_sim: int = 64,
    vmin: float = 0.0,
    vmax: float = 0.2,
    num_workers: int = 7,
):
    """
    Multi-family physics-based evaluation (anisotropic), NO language input.

    For each trajectory (family, sim_dir), we:
      - predict a_hat_x(t), a_hat_y(t) at all intervals k
      - re-simulate φ_hat with appropriate anisotropic diffusion type
      - compare to ground-truth φ_true
    """
    model.eval()

    if isinstance(ds_or_subset, Subset):
        base_ds = ds_or_subset.dataset
        idx_list = ds_or_subset.indices
    else:
        base_ds = ds_or_subset
        idx_list = list(range(len(base_ds)))

    assert isinstance(base_ds, MultiFamilyCalibDiffDataset), \
        "eval_epoch_pde_multi expects MultiFamilyCalibDiffDataset"

    # collect unique trajectories (family, sim_idx)
    traj_keys = set()
    for idx in idx_list:
        fam, sim_idx, k = base_ds.index[idx]
        traj_keys.add((fam, sim_idx))

    traj_keys = sorted(list(traj_keys))
    if max_sims is not None:
        traj_keys = traj_keys[:max_sims]

    img_size = base_ds.img_size
    tasks = []

    # -------- Stage 1: predict a_hat_x/y(t) for each trajectory --------
    for fam, sim_idx in traj_keys:
        fam_info = base_ds.family_info[fam]
        sim_dir = fam_info["sim_dirs"][sim_idx]
        K = fam_info["K"]

        with open(os.path.join(sim_dir, "params.json"), "r") as f:
            params = json.load(f)

        ax_hat_list = []
        ay_hat_list = []

        for k in range(K):
            phi_k_path = os.path.join(sim_dir, f"phi_interval_{k:02d}.png")
            phi_goal_path = os.path.join(sim_dir, f"phi_interval_{K:02d}.png")

            I_k = base_ds.transform(Image.open(phi_k_path).convert("RGB"))
            I_goal = base_ds.transform(Image.open(phi_goal_path).convert("RGB"))
            tau_k = torch.tensor([k / (K - 1)], dtype=torch.float32)

            I_k = I_k.unsqueeze(0).to(device)
            I_goal = I_goal.unsqueeze(0).to(device)
            tau_k = tau_k.unsqueeze(0).to(device)

            logits_x, offsets_x, logits_y, offsets_y = model(
                I_k, I_goal, tau_k
            )  # [1,n_bins] each

            bins_x = torch.argmax(logits_x, dim=-1)  # [1]
            bins_y = torch.argmax(logits_y, dim=-1)  # [1]

            a_hat = model.bins_to_param_2d(
                bins_x, offsets_x, bins_y, offsets_y
            )[0].detach().cpu().numpy()  # [2]

            ax_hat_list.append(a_hat[0])
            ay_hat_list.append(a_hat[1])

        ax_hat_array = np.array(ax_hat_list, dtype=float)
        ay_hat_array = np.array(ay_hat_list, dtype=float)

        tasks.append(
            (
                fam,
                sim_dir,
                params,
                ax_hat_array,
                ay_hat_array,
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

    # -------- Stage 2: PDE sims in parallel --------
    total_rgb_mse = 0.0
    total_conc_mse = 0.0
    successful_recoveries = 0
    n_sims_used = 0

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        for conc_mse, rgb_mse, param_success in ex.map(_multi_aniso_worker_task, tasks):
            total_conc_mse += conc_mse
            total_rgb_mse += rgb_mse
            successful_recoveries += int(param_success)
            n_sims_used += 1

    denom = max(n_sims_used, 1)
    mean_rgb_mse = total_rgb_mse / denom
    mean_conc_mse = total_conc_mse / denom
    param_recovery_rate = successful_recoveries / denom

    return mean_rgb_mse, mean_conc_mse, param_recovery_rate


# -------------------------------------------------------------------------
# 4. Training & validation loops (anisotropic, NO language)
# -------------------------------------------------------------------------

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
        target_params = batch["target_params"].to(device)  # [B,2]

        logits_x, offsets_x, logits_y, offsets_y = model(
            I_k, I_goal, tau_k
        )  # [B,n_bins] each

        target_bins_x, target_bins_y = model.discretize_2d(target_params)  # [B],[B]

        loss_cls_x = F.cross_entropy(logits_x, target_bins_x)
        loss_cls_y = F.cross_entropy(logits_y, target_bins_y)
        loss_cls = 0.5 * (loss_cls_x + loss_cls_y)

        # regression target: offsets for x and y
        bin_width = (model.theta_max - model.theta_min) / model.n_bins

        bin_centers_x = model.theta_min + (
            (target_bins_x.float() + 0.5) / model.n_bins
        ) * (model.theta_max - model.theta_min)

        bin_centers_y = model.theta_min + (
            (target_bins_y.float() + 0.5) / model.n_bins
        ) * (model.theta_max - model.theta_min)

        offset_target_x = (target_params[:, 0] - bin_centers_x) / bin_width
        offset_target_y = (target_params[:, 1] - bin_centers_y) / bin_width

        offset_target_x = torch.clamp(offset_target_x, -0.5, 0.5)
        offset_target_y = torch.clamp(offset_target_y, -0.5, 0.5)

        offset_pred_x = offsets_x.gather(1, target_bins_x.view(-1, 1)).squeeze(1)
        offset_pred_y = offsets_y.gather(1, target_bins_y.view(-1, 1)).squeeze(1)

        loss_reg_x = F.mse_loss(offset_pred_x, offset_target_x)
        loss_reg_y = F.mse_loss(offset_pred_y, offset_target_y)
        loss_reg = 0.5 * (loss_reg_x + loss_reg_y)

        loss = loss_cls + 0.1 * loss_reg

        total_loss += loss.item()
        total_loss_cls += loss_cls.item()
        total_loss_reg += loss_reg.item()
        n_batches += 1

    n_batches = max(n_batches, 1)
    return total_loss / n_batches


def train_epoch(
    model: CalibDiff,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_every: int = 50,
    curve_path: str = "train_loss_curve_multi_ani_nolang.png",
):
    model.train()
    total_loss = 0.0
    n_batches = 0

    loss_history = []

    for step, batch in enumerate(dataloader, start=1):
        I_k = batch["I_k"].to(device)
        I_goal = batch["I_goal"].to(device)
        tau_k = batch["tau_k"].to(device)
        target_params = batch["target_params"].to(device)  # [B,2]

        optimizer.zero_grad()

        logits_x, offsets_x, logits_y, offsets_y = model(
            I_k, I_goal, tau_k
        )

        target_bins_x, target_bins_y = model.discretize_2d(target_params)

        loss_cls_x = F.cross_entropy(logits_x, target_bins_x)
        loss_cls_y = F.cross_entropy(logits_y, target_bins_y)
        loss_cls = 0.5 * (loss_cls_x + loss_cls_y)

        with torch.no_grad():
            bin_width = (model.theta_max - model.theta_min) / model.n_bins

            bin_centers_x = model.theta_min + (
                (target_bins_x.float() + 0.5) / model.n_bins
            ) * (model.theta_max - model.theta_min)

            bin_centers_y = model.theta_min + (
                (target_bins_y.float() + 0.5) / model.n_bins
            ) * (model.theta_max - model.theta_min)

            offset_target_x = (target_params[:, 0] - bin_centers_x) / bin_width
            offset_target_y = (target_params[:, 1] - bin_centers_y) / bin_width

            offset_target_x = torch.clamp(offset_target_x, -0.5, 0.5)
            offset_target_y = torch.clamp(offset_target_y, -0.5, 0.5)

        offset_pred_x = offsets_x.gather(1, target_bins_x.view(-1, 1)).squeeze(1)
        offset_pred_y = offsets_y.gather(1, target_bins_y.view(-1, 1)).squeeze(1)

        loss_reg_x = F.mse_loss(offset_pred_x, offset_target_x)
        loss_reg_y = F.mse_loss(offset_pred_y, offset_target_y)
        loss_reg = 0.5 * (loss_reg_x + loss_reg_y)

        loss = loss_cls + 0.1 * loss_reg
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        loss_history.append(loss_value)

        total_loss += loss_value
        n_batches += 1

        if step % log_every == 0:
            stride = 10
            indices = list(range(0, len(loss_history), stride))
            steps_ds = [i + 1 for i in indices]
            loss_ds = [loss_history[i] for i in indices]

            plt.figure()
            plt.plot(steps_ds, loss_ds, marker="o", linewidth=1)
            plt.xlabel("Batch index (within epoch)")
            plt.ylabel("Training loss")
            plt.title("Training loss curve (within epoch, downsampled)")
            plt.grid(True, linestyle="--", alpha=0.3)
            plt.savefig(curve_path, dpi=150, bbox_inches="tight")
            plt.close()

    return total_loss / max(n_batches, 1)


def plot_all_losses(
    train_losses,
    val_losses,
    pde_losses,
    out_path: str = "loss_curves_all_multi_ani_nolang.png",
):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Val loss")
    plt.plot(epochs, pde_losses, label="PDE RGB MSE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / MSE")
    plt.title("Training / Validation / PDE Loss Curves (multi anisotropic, NO language)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# -------------------------------------------------------------------------
# 5. Main: multi-family anisotropic training + evaluation (NO language)
# -------------------------------------------------------------------------

def main():
    # --- Configure which anisotropic families participate & where their data live ---
    family_roots = {
        "lin_ani": "/data/IMcali/sim_results_2025-12-19 04:48:56_lin_ani",
        "exp_ani": "/data/IMcali/sim_results_2025-12-19 18:26:15_exp_ani",
    }

    img_size = 128
    patch_size = 2
    batch_size = 16
    num_epochs = 100

    # global parameter range shared by all anisotropic families
    theta_min = 1.0
    theta_max = 10.0
    n_bins = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # ----------------------------------------------------------
    # Build dataset
    # ----------------------------------------------------------
    dataset = MultiFamilyCalibDiffDataset(
        family_roots=family_roots,
        img_size=img_size,
    )

    # split over all (family, sim_idx, k) samples
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

    # ----------------------------------------------------------
    # Model (NO language)
    # ----------------------------------------------------------
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

        pde_rgb_mse, conc_mse, param_rec = eval_epoch_pde_multi(
            model,
            val_set,
            device,
            max_sims=50,
            nx_sim=64,
            ny_sim=64,
        )

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
            out_path="loss_curves_all_multi_ani_nolang.png",
        )

    torch.save(model.state_dict(), "calibdiff_multi_ani_nolang_checkpoint.pt")
    print("Training complete, model saved to calibdiff_multi_ani_nolang_checkpoint.pt")

    np.savez(
        "loss_histories_all_multi_ani_nolang.npz",
        train_loss=train_loss_hist,
        val_loss=val_loss_hist,
        pde_rgb_mse=pde_rgb_mse_hist,
        conc_mse=conc_mse_hist,
        param_recovery=param_rec_hist,
    )


if __name__ == "__main__":
    main()
