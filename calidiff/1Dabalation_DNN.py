import os
import json
import glob
from typing import Dict, List, Tuple
import random  # <-- NEW

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
# External diffusion-family modules (your existing single-family scripts)
# -------------------------------------------------------------------------
import cali1_iso1_lin_iso as lin_iso_mod
import cali1_iso1_exp_iso as exp_iso_mod
import cali1_iso1_concrete as concrete_mod

# -------------------------------------------------------------------------
# CLIP imports
# -------------------------------------------------------------------------
import clip


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
# Per-family parameter ranges (raw, physical space)
# -------------------------------------------------------------------------
FAMILY_THETA: Dict[str, Tuple[float, float]] = {
    "lin_iso":   (1.0, 10.0),  # a(t) in [1, 10]
    "exp_iso":   (1.0, 10.0),  # a(t) in [1, 10]
    "concrete":  (0.1, 1.0),   # A0(t) in [0.1, 1.0]
}

# Global training/discretization range (shared by all families)
GLOBAL_THETA_MIN = 1.0
GLOBAL_THETA_MAX = 10.0


def scale_param_to_global(family: str, a_raw: float) -> float:
    """
    Map raw family-specific parameter a_raw in [theta_min_fam, theta_max_fam]
    to global range [GLOBAL_THETA_MIN, GLOBAL_THETA_MAX].
    """
    theta_min_fam, theta_max_fam = FAMILY_THETA[family]
    if theta_max_fam <= theta_min_fam:
        return GLOBAL_THETA_MIN
    t = (a_raw - theta_min_fam) / (theta_max_fam - theta_min_fam)
    return GLOBAL_THETA_MIN + t * (GLOBAL_THETA_MAX - GLOBAL_THETA_MIN)


def unscale_param_from_global(family: str, a_global: float) -> float:
    """
    Map global parameter a_global in [GLOBAL_THETA_MIN, GLOBAL_THETA_MAX]
    back to raw family-specific range [theta_min_fam, theta_max_fam].
    """
    theta_min_fam, theta_max_fam = FAMILY_THETA[family]
    if GLOBAL_THETA_MAX <= GLOBAL_THETA_MIN:
        return theta_min_fam
    t = (a_global - GLOBAL_THETA_MIN) / (GLOBAL_THETA_MAX - GLOBAL_THETA_MIN)
    return theta_min_fam + t * (theta_max_fam - theta_min_fam)


# -------------------------------------------------------------------------
# 1. Multi-family dataset (1D parameter, multiple diffusion families)
# -------------------------------------------------------------------------

class MultiFamilyCalibDiffDataset(Dataset):
    """
    Mixed dataset over multiple diffusion 'families', all with a single
    scalar parameter per time interval (a(t) or A0(t)) stored in
    params["coefficients_ax"].

    The *raw* parameters differ per family:
        lin_iso, exp_iso: [1, 10]
        concrete: [0.1, 1.0]

    In this dataset, we SCALE every raw parameter into the global range
    [GLOBAL_THETA_MIN, GLOBAL_THETA_MAX] before feeding to the model.
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
                    "lin_iso": "/data/..._lin_iso",
                    "exp_iso": "/data/..._exp_iso",
                    "concrete": "/data/..._concrete_iso",
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

        # For each family: collect sim dirs, read K from first params.json
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

            self.family_info[fam] = {
                "root_dir": root,
                "sim_dirs": sim_dirs,
                "K": K,
            }

            for sim_idx in range(len(sim_dirs)):
                for k in range(K):
                    self.items.append((fam, sim_idx, k))

        # this mirrors your per-family dataset "index" concept
        self.index = self.items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fam, sim_idx, k = self.items[idx]
        fam_info = self.family_info[fam]
        sim_dir = fam_info["sim_dirs"][sim_idx]
        K = fam_info["K"]

        # params.json holds coefficients_ax; raw, family-specific scale
        with open(os.path.join(sim_dir, "params.json"), "r") as f:
            params = json.load(f)

        a_k_raw = float(params["coefficients_ax"][k])

        # ---- SCALE raw param to global range [GLOBAL_THETA_MIN, GLOBAL_THETA_MAX]
        a_k_global = scale_param_to_global(fam, a_k_raw)
        target = torch.tensor([a_k_global], dtype=torch.float32)   # [1]

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
            "target_params": target,   # [1], in *global* θ range
            "family": fam,
            "sim_idx": sim_idx,
            "k": k,
        }


# -------------------------------------------------------------------------
# 2. CalibDiff model with CLIP language conditioning
#    ABLATION: replace PerceiverIO-style attention bottleneck with a deep DNN
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


class ConvResBlock(nn.Module):
    """
    Simple deep neural network block (no attention):
      GN -> GELU -> Conv -> GN -> GELU -> Conv + residual
    """
    def __init__(self, ch: int, gn_groups: int = 8):
        super().__init__()
        g = min(gn_groups, ch)
        while ch % g != 0 and g > 1:
            g -= 1

        self.gn1 = nn.GroupNorm(g, ch)
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=2, padding=0)
        self.gn2 = nn.GroupNorm(g, ch)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.gelu(self.gn1(x)))
        h = self.conv2(F.gelu(self.gn2(h)))
        return x + h


class CalibDiff(nn.Module):
    def __init__(
        self,
        img_size=128,
        patch_size=8,
        embed_dim=128,
        num_heads=4,         # kept for API-compat (unused in this ablation)
        num_latents=256,     # reused as width hint
        num_latent_layers=4, # reused as depth hint
        theta_min=GLOBAL_THETA_MIN,      # global range across families (for *scaled* θ)
        theta_max=GLOBAL_THETA_MAX,
        n_bins=8,            # number of classification bins
        lang_dim: int = 512,
    ):
        """
        CalibDiff with an extra CLIP language token.

        ABLATION:
          - The PerceiverIO latent bottleneck (cross-attn + latent transformer)
            is replaced by a deep convolutional residual network operating on
            patch grids, with time/lang conditioning broadcast over space.

        theta_min/theta_max are the *global scaled* range shared by all families.
        lang_dim: dimension of CLIP text embeddings.
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.n_bins = n_bins
        self.lang_dim = lang_dim
        self.use_lang = lang_dim is not None and lang_dim > 0

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

        # Modality embeddings: current vs goal (token-wise, same as before)
        self.mod_cur = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mod_goal = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Time token projection (will be broadcast over spatial grid)
        self.time_token_proj = nn.Linear(1, embed_dim)

        # Language projection: CLIP text embedding -> model dimension (broadcast)
        if self.use_lang:
            self.lang_proj = nn.Linear(lang_dim, embed_dim)

        # Learnable 2D positional embedding on the patch grid (for spatial DNN)
        self.pos2d = nn.Parameter(torch.randn(1, embed_dim, H_p, W_p) * 0.01)

        # -----------------------------------------------------------------
        # Deep neural network fusion trunk (no attention)
        # -----------------------------------------------------------------
        # Reuse num_latents as a width hint (clamped), and num_latent_layers as depth hint.
        width_mult = max(1, int(round(num_latents / 256)))
        hidden_ch = min(embed_dim * width_mult, 1024)  # safety clamp

        in_ch = 2 * embed_dim + embed_dim  # cur + goal + time
        if self.use_lang:
            in_ch += embed_dim             # + lang

        self.fuse_in = nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1)

        # Make it "deep": use ~2x the old latent depth by default
        conv_depth = 2
        self.trunk = nn.Sequential(*[ConvResBlock(hidden_ch) for _ in range(conv_depth)])

        self.fuse_out_norm = nn.GroupNorm(
            num_groups=max(1, min(8, hidden_ch)),
            num_channels=hidden_ch
        )

        # Head: logits over bins + per-bin offsets (same interface as original)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_ch),
            nn.Linear(hidden_ch, 2 * n_bins),
        )

    def forward(self, I_k, I_goal, tau_k, lang_emb):
        """
        Args:
            I_k:      [B,3,H,W]
            I_goal:   [B,3,H,W]
            tau_k:    [B,1] normalized time index
            lang_emb: [B,lang_dim] CLIP text embedding for the family description
        Returns:
            logits:  [B, n_bins]
            offsets: [B, n_bins]
        """
        B = I_k.shape[0]

        # patch embeddings (tokens)
        z_cur, H_p, W_p = self.patch_embed(I_k)   # [B,N_p,C]
        z_goal, _, _ = self.patch_embed(I_goal)   # [B,N_p,C]
        assert H_p == self.H_p and W_p == self.W_p

        # modality embeddings (token domain, same as original)
        z_cur = z_cur + self.mod_cur
        z_goal = z_goal + self.mod_goal

        # reshape tokens -> feature maps for deep DNN
        f_cur = z_cur.transpose(1, 2).reshape(B, self.embed_dim, H_p, W_p)   # [B,C,H_p,W_p]
        f_goal = z_goal.transpose(1, 2).reshape(B, self.embed_dim, H_p, W_p) # [B,C,H_p,W_p]

        # add learnable spatial positional bias
        f_cur = f_cur + self.pos2d
        f_goal = f_goal + self.pos2d

        # time conditioning (broadcast to spatial grid)
        f_time = self.time_token_proj(tau_k)        # [B,C]
        f_time = f_time.unsqueeze(-1).unsqueeze(-1) # [B,C,1,1]
        f_time = f_time.expand(-1, -1, H_p, W_p)    # [B,C,H_p,W_p]

        # language conditioning (broadcast)
        feats = [f_cur, f_goal, f_time]
        if self.use_lang:
            f_lang = self.lang_proj(lang_emb)           # [B,C]
            f_lang = f_lang.unsqueeze(-1).unsqueeze(-1) # [B,C,1,1]
            f_lang = f_lang.expand(-1, -1, H_p, W_p)    # [B,C,H_p,W_p]
            feats.append(f_lang)

        x = torch.cat(feats, dim=1)   # [B, in_ch, H_p, W_p]

        # deep fusion trunk
        x = self.fuse_in(x)
        x = self.trunk(x)
        x = F.gelu(self.fuse_out_norm(x))

        # global average pool
        z = x.mean(dim=(2, 3))        # [B, hidden_ch]

        head_out = self.head(z)       # [B,2*n_bins]
        logits, offset_raw = head_out.chunk(2, dim=-1)
        offsets = 0.5 * torch.tanh(offset_raw)   # [-0.5,0.5] in bin units
        return logits, offsets

    # ---------------- 1D discretization helpers -----------------

    def discretize_1d(self, params: torch.Tensor):
        """
        Convert continuous params [B,1] in *global* θ-space
        into bin indices [0..n_bins-1].
        """
        a = params[:, 0]  # [B]

        v_clamp = torch.clamp(a, self.theta_min, self.theta_max)
        ratio = (v_clamp - self.theta_min) / max(self.theta_max - self.theta_min, 1e-6)
        idx = torch.round(ratio * (self.n_bins - 1)).long()
        idx = torch.clamp(idx, 0, self.n_bins - 1)
        return idx

    def bins_to_param(self, bin_idx: torch.Tensor, offsets: torch.Tensor):
        """
        Given bin indices [B] and per-bin offsets [B,n_bins],
        map to a continuous scalar a_hat [B] in *global* θ-space.
        """
        bin_idx_long = bin_idx.long()
        B = bin_idx_long.shape[0]

        offset_sel = offsets.gather(1, bin_idx_long.view(B, 1)).squeeze(1)  # [B]

        bin_idx_f = bin_idx_long.float()
        bin_width = (self.theta_max - self.theta_min) / self.n_bins
        bin_centers = self.theta_min + (bin_idx_f + 0.5) / self.n_bins * (
            self.theta_max - self.theta_min
        )

        a_hat = bin_centers + offset_sel * bin_width
        return a_hat


# -------------------------------------------------------------------------
# 3. Multi-family PDE worker & evaluation
# -------------------------------------------------------------------------

def _multi_pde_worker_task(args):
    """
    Runs in a separate CPU process:
      - simulate φ_hat with predicted a_hat(t) (raw or A0_hat(t) for concrete)
      - simulate φ_true with ground-truth a_true(t) / A0_true(t)
      - compute MSEs and parameter recovery flag
    """
    (
        family,
        sim_dir,
        params,
        a_hat_array_raw,  # <-- now RAW, family-specific range
        K,
        nx_sim,
        ny_sim,
        img_size,
        vmin,
        vmax,
        theta_min_fam,
        theta_max_fam,
    ) = args

    # ground truth (raw)
    a_true_array = np.array(params["coefficients_ax"], dtype=float)[:K]

    # choose PDE simulator (expects RAW parameters)
    if family == "lin_iso":
        phi_hat = lin_iso_mod.simulate_final_phi_isotropic(
            params, a_hat_array_raw, nx=nx_sim, ny=ny_sim
        )
        phi_true = lin_iso_mod.simulate_final_phi_isotropic(
            params, a_true_array, nx=nx_sim, ny=ny_sim
        )
    elif family == "exp_iso":
        phi_hat = exp_iso_mod.simulate_final_phi_isotropic(
            params, a_hat_array_raw, nx=nx_sim, ny=ny_sim
        )
        phi_true = exp_iso_mod.simulate_final_phi_isotropic(
            params, a_true_array, nx=nx_sim, ny=ny_sim
        )
    elif family == "concrete":
        phi_hat = concrete_mod.simulate_final_phi_chloride(
            params, a_hat_array_raw, nx=nx_sim, ny=ny_sim
        )
        phi_true = concrete_mod.simulate_final_phi_chloride(
            params, a_true_array, nx=nx_sim, ny=ny_sim
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

    # parameter recovery: 10% of *family-specific* calibration range in RAW space
    theta_range = theta_max_fam - theta_min_fam
    param_tol = 0.1 * theta_range

    param_success = bool(np.all(np.abs(a_hat_array_raw - a_true_array) < param_tol))

    return conc_mse, rgb_mse, param_success


@torch.no_grad()
def eval_epoch_pde_multi(
    model: CalibDiff,
    ds_or_subset,
    device: torch.device,
    fam2lang: Dict[str, torch.Tensor],
    max_sims: int = 50,
    nx_sim: int = 64,
    ny_sim: int = 64,
    vmin: float = 0.0,
    vmax: float = 0.2,
    num_workers: int = 7,
):
    """
    Multi-family physics-based evaluation.

    For each trajectory (family, sim_dir), we:
      - predict a_hat_global(t) at all intervals k (with language conditioning)
      - UNSCALE a_hat_global(t) to raw family-specific a_hat_raw(t)
      - re-simulate φ_hat with appropriate diffusion type
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

    traj_keys = list(traj_keys)

    # RANDOM sampling of trajectories across families (instead of sorted slice)
    if max_sims is not None and len(traj_keys) > max_sims:
        traj_keys = random.sample(traj_keys, max_sims)

    img_size = base_ds.img_size

    tasks = []

    # -------- Stage 1: predict a_hat(t) for each trajectory (GLOBAL θ) --------
    for fam, sim_idx in traj_keys:
        fam_info = base_ds.family_info[fam]
        sim_dir = fam_info["sim_dirs"][sim_idx]
        K = fam_info["K"]

        with open(os.path.join(sim_dir, "params.json"), "r") as f:
            params = json.load(f)

        a_hat_list_raw = []  # will store RAW params (family-specific)

        # CLIP language embedding for this family
        lang_emb_fam = fam2lang[fam].unsqueeze(0).to(device)  # [1,lang_dim]

        for k in range(K):
            phi_k_path = os.path.join(sim_dir, f"phi_interval_{k:02d}.png")
            phi_goal_path = os.path.join(sim_dir, f"phi_interval_{K:02d}.png")

            I_k = base_ds.transform(Image.open(phi_k_path).convert("RGB"))
            I_goal = base_ds.transform(Image.open(phi_goal_path).convert("RGB"))
            tau_k = torch.tensor([k / (K - 1)], dtype=torch.float32)

            I_k = I_k.unsqueeze(0).to(device)
            I_goal = I_goal.unsqueeze(0).to(device)
            tau_k = tau_k.unsqueeze(0).to(device)

            logits, offsets = model(I_k, I_goal, tau_k, lang_emb_fam)  # [1,n_bins],[1,n_bins]
            pred_bins = torch.argmax(logits, dim=-1)     # [1]

            # prediction in GLOBAL θ range
            a_hat_global = model.bins_to_param(pred_bins, offsets)[0].item()

            # UNSCALE to RAW family-specific parameter
            a_hat_raw = unscale_param_from_global(fam, a_hat_global)
            a_hat_list_raw.append(a_hat_raw)

        a_hat_array_raw = np.array(a_hat_list_raw, dtype=float)

        # family-specific theta range for parameter recovery check (RAW space)
        theta_min_fam, theta_max_fam = FAMILY_THETA[fam]

        tasks.append(
            (
                fam,
                sim_dir,
                params,
                a_hat_array_raw,
                K,
                nx_sim,
                ny_sim,
                img_size,
                vmin,
                vmax,
                theta_min_fam,
                theta_max_fam,
            )
        )

    # -------- Stage 2: PDE sims in parallel --------
    total_rgb_mse = 0.0
    total_conc_mse = 0.0
    successful_recoveries = 0
    n_sims_used = 0

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        for conc_mse, rgb_mse, param_success in ex.map(_multi_pde_worker_task, tasks):
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
# 4. Training & validation loops (same loss as your 1D scripts, + language)
# -------------------------------------------------------------------------

def _build_lang_batch(
    family_batch: List[str],
    fam2lang: Dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """
    Convert a list of family names into a [B, lang_dim] tensor
    of CLIP language embeddings.
    """
    emb_list = [fam2lang[fam] for fam in family_batch]
    lang_batch = torch.stack(emb_list, dim=0).to(device)  # [B,lang_dim]
    return lang_batch


@torch.no_grad()
def eval_epoch(
    model: CalibDiff,
    dataloader: DataLoader,
    device: torch.device,
    fam2lang: Dict[str, torch.Tensor],
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
        target_params = batch["target_params"].to(device)  # [B,1], GLOBAL θ
        family_batch = batch["family"]  # list of strings length B

        lang_batch = _build_lang_batch(family_batch, fam2lang, device)

        logits, offsets = model(I_k, I_goal, tau_k, lang_batch)  # [B,n_bins],[B,n_bins]

        target_bins = model.discretize_1d(target_params)
        loss_cls = F.cross_entropy(logits, target_bins)

        bin_idx_f = target_bins.float()
        bin_width = (model.theta_max - model.theta_min) / model.n_bins
        bin_centers = model.theta_min + (bin_idx_f + 0.5) / model.n_bins * (
            model.theta_max - model.theta_min
        )
        offset_target = (target_params.squeeze(1) - bin_centers) / bin_width
        offset_target = torch.clamp(offset_target, -0.5, 0.5)

        offset_pred = offsets.gather(1, target_bins.view(-1, 1)).squeeze(1)
        loss_reg = F.mse_loss(offset_pred, offset_target)

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
    fam2lang: Dict[str, torch.Tensor],
    log_every: int = 50,
    curve_path: str = "train_loss_curve_multi_iso_concrete_clip.png",
):
    model.train()
    total_loss = 0.0
    n_batches = 0

    loss_history = []

    for step, batch in enumerate(dataloader, start=1):
        I_k = batch["I_k"].to(device)
        I_goal = batch["I_goal"].to(device)
        tau_k = batch["tau_k"].to(device)
        target_params = batch["target_params"].to(device)  # [B,1], GLOBAL θ
        family_batch = batch["family"]  # list of strings

        lang_batch = _build_lang_batch(family_batch, fam2lang, device)

        optimizer.zero_grad()

        logits, offsets = model(I_k, I_goal, tau_k, lang_batch)

        target_bins = model.discretize_1d(target_params)
        loss_cls = F.cross_entropy(logits, target_bins)

        with torch.no_grad():
            bin_idx_f = target_bins.float()
            bin_width = (model.theta_max - model.theta_min) / model.n_bins
            bin_centers = model.theta_min + (bin_idx_f + 0.5) / model.n_bins * (
                model.theta_max - model.theta_min
            )
            offset_target = (target_params.squeeze(1) - bin_centers) / bin_width
            offset_target = torch.clamp(offset_target, -0.5, 0.5)

        offset_pred = offsets.gather(1, target_bins.view(-1, 1)).squeeze(1)
        loss_reg = F.mse_loss(offset_pred, offset_target)

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
    out_path: str = "loss_curves_all_multi_iso_concrete_clip.png",
):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Val loss")
    plt.plot(epochs, pde_losses, label="PDE RGB MSE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / MSE")
    plt.title("Training / Validation / PDE Loss Curves (multi-family + CLIP)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# -------------------------------------------------------------------------
# 5. Main: multi-family / cross-family training + evaluation with CLIP
# -------------------------------------------------------------------------

def main():
    # --- Configure which families participate & where their data live ---
    family_roots = {
        "lin_iso":   "/data/IMcali/sim_results_2025-12-17 22:31:04_lin_iso",
        "exp_iso":   "/data/IMcali/sim_results_2025-12-19 18:25:27_exp_iso",
        "concrete":  "/data/IMcali/sim_results_2025-12-19 15:52:24_concrete_iso",
    }

    # Text descriptions per family (language supervision)
    family_text = {
        "exp_iso":  "exponential isotropic diffusion over four equal time intervals",
        "lin_iso":  "linear isotropic diffusion over four equal time intervals",
        "concrete": "chloride diffusion into concrete over four equal time intervals",
    }

    img_size = 128
    patch_size = 2
    batch_size = 16
    num_epochs = 100

    # global *scaled* parameter range that all families share
    theta_min = GLOBAL_THETA_MIN
    theta_max = GLOBAL_THETA_MAX
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

    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    fam2lang: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for fam, text in family_text.items():
            tokens = clip.tokenize([text]).to(device)  # [1, seq_len]
            sentence_emb = clip_model.encode_text(tokens)
            sentence_emb = sentence_emb / sentence_emb.norm(dim=-1, keepdim=True)
            sentence_emb = sentence_emb.to(dtype=torch.float32)
            fam2lang[fam] = sentence_emb.squeeze(0).detach().cpu()

    # infer CLIP embedding dimension
    lang_dim = next(iter(fam2lang.values())).shape[-1]

    # ----------------------------------------------------------
    # Model
    # ----------------------------------------------------------
    model = CalibDiff(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=128,
        num_heads=4,            # unused in this ablation (kept for identical call)
        num_latents=2048,       # reused as width hint
        num_latent_layers=6,    # reused as depth hint
        theta_min=theta_min,
        theta_max=theta_max,
        n_bins=n_bins,
        lang_dim=lang_dim,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    train_loss_hist = []
    val_loss_hist = []
    pde_rgb_mse_hist = []
    conc_mse_hist = []
    param_rec_hist = []

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, fam2lang)
        val_loss = eval_epoch(model, val_loader, device, fam2lang)

        pde_rgb_mse, conc_mse, param_rec = eval_epoch_pde_multi(
            model,
            val_set,
            device,
            fam2lang,
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
            out_path="loss_curves_all_multi_iso_concrete_clip.png",
        )

    torch.save(model.state_dict(), "calibdiff_multi_iso_concrete_clip_checkpoint.pt")
    print("Training complete, model saved to calibdiff_multi_iso_concrete_clip_checkpoint.pt")

    np.savez(
        "loss_histories_all_multi_iso_concrete_clip.npz",
        train_loss=train_loss_hist,
        val_loss=val_loss_hist,
        pde_rgb_mse=pde_rgb_mse_hist,
        conc_mse=conc_mse_hist,
        param_recovery=param_rec_hist,
    )


if __name__ == "__main__":
    main()
