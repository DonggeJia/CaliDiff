import os
import json
import glob
from typing import Dict, List, Tuple
import random

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
        super().__init__()
        self.family_roots = dict(family_roots)
        self.img_size = img_size

        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),          # [0,1], [C,H,W]
        ])

        self.family_info: Dict[str, Dict] = {}
        self.items: List[Tuple[str, int, int]] = []  # (family, sim_idx, k)

        for fam, root in self.family_roots.items():
            sim_dirs = sorted(
                d for d in glob.glob(os.path.join(root, "sim_*"))
                if os.path.isdir(d)
            )
            if len(sim_dirs) == 0:
                raise ValueError(f"No sim_* folders found under {root} for family '{fam}'")

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

        self.index = self.items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fam, sim_idx, k = self.items[idx]
        fam_info = self.family_info[fam]
        sim_dir = fam_info["sim_dirs"][sim_idx]
        K = fam_info["K"]

        with open(os.path.join(sim_dir, "params.json"), "r") as f:
            params = json.load(f)

        a_k_raw = float(params["coefficients_ax"][k])
        a_k_global = scale_param_to_global(fam, a_k_raw)
        target = torch.tensor([a_k_global], dtype=torch.float32)   # [1]

        phi_k_path = os.path.join(sim_dir, f"phi_interval_{k:02d}.png")
        phi_goal_path = os.path.join(sim_dir, f"phi_interval_{K:02d}.png")

        I_k = self.transform(Image.open(phi_k_path).convert("RGB"))       # [3,H,W]
        I_goal = self.transform(Image.open(phi_goal_path).convert("RGB")) # [3,H,W]

        tau_k = torch.tensor([k / (K - 1)], dtype=torch.float32)

        return {
            "I_k": I_k,
            "I_goal": I_goal,
            "tau_k": tau_k,
            "target_params": target,   # [1], GLOBAL θ range
            "family": fam,
            "sim_idx": sim_idx,
            "k": k,
        }


# -------------------------------------------------------------------------
# 2. CalibDiff model with CLIP language conditioning
#    ABLATION: continuous scalar predictor (no bins + offsets)
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
        x = self.conv(x)
        B, C, H_p, W_p = x.shape
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
        q = self.norm_q(query)
        k = self.norm_ctx(context)
        v = k
        out, _ = self.attn(q, k, v)
        return query + out


class CalibDiffContinuous(nn.Module):
    def __init__(
        self,
        img_size=128,
        patch_size=8,
        embed_dim=128,
        num_heads=4,
        num_latents=256,
        num_latent_layers=4,
        theta_min=GLOBAL_THETA_MIN,
        theta_max=GLOBAL_THETA_MAX,
        lang_dim: int = 512,
    ):
        """
        Continuous ablation: predict a single scalar \hat{a} in GLOBAL θ-space.

        We produce an unconstrained scalar, then squash with sigmoid into
        [theta_min, theta_max] for stability.
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.theta_min = float(theta_min)
        self.theta_max = float(theta_max)
        self.lang_dim = lang_dim
        self.use_lang = lang_dim is not None and lang_dim > 0

        self.patch_embed = PatchEmbed(
            in_ch=3, embed_dim=embed_dim, patch_size=patch_size, img_size=img_size
        )

        H_p = img_size // patch_size
        W_p = img_size // patch_size
        self.H_p = H_p
        self.W_p = W_p
        self.N_p = H_p * W_p

        self.mod_cur = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mod_goal = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.time_token_proj = nn.Linear(1, embed_dim)

        if self.use_lang:
            self.lang_proj = nn.Linear(lang_dim, embed_dim)

        total_tokens = 2 * self.N_p + 1 + (1 if self.use_lang else 0)
        self.total_tokens = total_tokens
        self.pos_embed = nn.Parameter(torch.randn(total_tokens, embed_dim) * 0.01)

        self.latents = nn.Parameter(torch.randn(1, num_latents, embed_dim) * 0.02)
        self.cross_attn_in = CrossAttention(embed_dim, num_heads)
        self.latent_blocks = nn.ModuleList(
            [LatentBlock(embed_dim, num_heads) for _ in range(num_latent_layers)]
        )
        self.cross_attn_out = CrossAttention(embed_dim, num_heads)

        # Continuous head
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),  # raw scalar
        )

    def forward(self, I_k, I_goal, tau_k, lang_emb):
        """
        Returns:
            a_hat_global: [B,1] in GLOBAL θ-space, squashed into [theta_min, theta_max]
        """
        B = I_k.shape[0]

        z_cur, H_p, W_p = self.patch_embed(I_k)   # [B,N_p,C]
        z_goal, _, _ = self.patch_embed(I_goal)   # [B,N_p,C]
        assert H_p == self.H_p and W_p == self.W_p

        z_cur = z_cur + self.mod_cur
        z_goal = z_goal + self.mod_goal
        z_patches = torch.cat([z_cur, z_goal], dim=1)  # [B,2N_p,C]

        z_time = self.time_token_proj(tau_k).unsqueeze(1)  # [B,1,C]

        if self.use_lang:
            z_lang = self.lang_proj(lang_emb).unsqueeze(1)  # [B,1,C]
            z0 = torch.cat([z_patches, z_time, z_lang], dim=1)
        else:
            z0 = torch.cat([z_patches, z_time], dim=1)

        N = z0.shape[1]
        assert N == self.total_tokens, f"Expected {self.total_tokens} tokens, got {N}"

        pos = self.pos_embed.unsqueeze(0).expand(B, -1, -1)
        z0 = z0 + pos

        latents = self.latents.expand(B, -1, -1)
        latents = self.cross_attn_in(latents, z0)
        for blk in self.latent_blocks:
            latents = blk(latents)
        zout = self.cross_attn_out(z0, latents)  # [B,N,C]

        z_cur_out = zout[:, : self.N_p, :]
        z_cur_pool = z_cur_out.mean(dim=1)       # [B,C]

        a_raw = self.head(z_cur_pool)            # [B,1], unconstrained
        a_sig = torch.sigmoid(a_raw)             # [0,1]
        a_hat = self.theta_min + a_sig * (self.theta_max - self.theta_min)  # [B,1]
        return a_hat


# -------------------------------------------------------------------------
# 3. Multi-family PDE worker & evaluation (continuous predictor)
# -------------------------------------------------------------------------

def _multi_pde_worker_task(args):
    (
        family,
        sim_dir,
        params,
        a_hat_array_raw,  # RAW, family-specific range
        K,
        nx_sim,
        ny_sim,
        img_size,
        vmin,
        vmax,
        theta_min_fam,
        theta_max_fam,
    ) = args

    a_true_array = np.array(params["coefficients_ax"], dtype=float)[:K]

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

    phi_hat_rgb_np = phi_to_rgb_colormap(phi_hat, vmin=vmin, vmax=vmax)
    phi_hat_img = Image.fromarray((phi_hat_rgb_np * 255).astype(np.uint8))
    phi_hat_img = phi_hat_img.resize((img_size, img_size), resample=Image.BILINEAR)
    phi_hat_rgb_resized = np.asarray(phi_hat_img).astype(np.float32) / 255.0

    goal_img = Image.open(os.path.join(sim_dir, f"phi_interval_{K:02d}.png")).convert("RGB")
    goal_img = goal_img.resize((img_size, img_size), resample=Image.BILINEAR)
    goal_np = np.asarray(goal_img).astype(np.float32) / 255.0

    rgb_mse = float(np.mean((phi_hat_rgb_resized - goal_np) ** 2))

    theta_range = theta_max_fam - theta_min_fam
    param_tol = 0.1 * theta_range
    param_success = bool(np.all(np.abs(a_hat_array_raw - a_true_array) < param_tol))

    return conc_mse, rgb_mse, param_success


@torch.no_grad()
def eval_epoch_pde_multi(
    model: CalibDiffContinuous,
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
    model.eval()

    if isinstance(ds_or_subset, Subset):
        base_ds = ds_or_subset.dataset
        idx_list = ds_or_subset.indices
    else:
        base_ds = ds_or_subset
        idx_list = list(range(len(base_ds)))

    assert isinstance(base_ds, MultiFamilyCalibDiffDataset), \
        "eval_epoch_pde_multi expects MultiFamilyCalibDiffDataset"

    traj_keys = set()
    for idx in idx_list:
        fam, sim_idx, k = base_ds.index[idx]
        traj_keys.add((fam, sim_idx))
    traj_keys = list(traj_keys)

    if max_sims is not None and len(traj_keys) > max_sims:
        traj_keys = random.sample(traj_keys, max_sims)

    img_size = base_ds.img_size
    tasks = []

    for fam, sim_idx in traj_keys:
        fam_info = base_ds.family_info[fam]
        sim_dir = fam_info["sim_dirs"][sim_idx]
        K = fam_info["K"]

        with open(os.path.join(sim_dir, "params.json"), "r") as f:
            params = json.load(f)

        a_hat_list_raw = []

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

            a_hat_global = model(I_k, I_goal, tau_k, lang_emb_fam)[0, 0].item()
            a_hat_raw = unscale_param_from_global(fam, a_hat_global)
            a_hat_list_raw.append(a_hat_raw)

        a_hat_array_raw = np.array(a_hat_list_raw, dtype=float)

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
# 4. Training & validation loops (continuous regression + language)
# -------------------------------------------------------------------------

def _build_lang_batch(
    family_batch: List[str],
    fam2lang: Dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    emb_list = [fam2lang[fam] for fam in family_batch]
    lang_batch = torch.stack(emb_list, dim=0).to(device)  # [B,lang_dim]
    return lang_batch


@torch.no_grad()
def eval_epoch(
    model: CalibDiffContinuous,
    dataloader: DataLoader,
    device: torch.device,
    fam2lang: Dict[str, torch.Tensor],
):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        I_k = batch["I_k"].to(device)
        I_goal = batch["I_goal"].to(device)
        tau_k = batch["tau_k"].to(device)
        target_params = batch["target_params"].to(device)  # [B,1], GLOBAL θ
        family_batch = batch["family"]  # list[str], length B

        lang_batch = _build_lang_batch(family_batch, fam2lang, device)

        pred_params = model(I_k, I_goal, tau_k, lang_batch)  # [B,1] GLOBAL θ
        loss = F.mse_loss(pred_params, target_params)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def train_epoch(
    model: CalibDiffContinuous,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    fam2lang: Dict[str, torch.Tensor],
    log_every: int = 50,
    curve_path: str = "train_loss_curve_multi_iso_concrete_clip_continuous.png",
):
    model.train()
    total_loss = 0.0
    n_batches = 0
    loss_history = []

    for step, batch in enumerate(dataloader, start=1):
        I_k = batch["I_k"].to(device)
        I_goal = batch["I_goal"].to(device)
        tau_k = batch["tau_k"].to(device)
        target_params = batch["target_params"].to(device)  # [B,1]
        family_batch = batch["family"]

        lang_batch = _build_lang_batch(family_batch, fam2lang, device)

        optimizer.zero_grad()
        pred_params = model(I_k, I_goal, tau_k, lang_batch)  # [B,1]
        loss = F.mse_loss(pred_params, target_params)
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
            plt.ylabel("Training MSE loss")
            plt.title("Training loss curve (within epoch, downsampled)")
            plt.grid(True, linestyle="--", alpha=0.3)
            plt.savefig(curve_path, dpi=150, bbox_inches="tight")
            plt.close()

    return total_loss / max(n_batches, 1)


def plot_all_losses(
    train_losses,
    val_losses,
    pde_losses,
    out_path: str = "loss_curves_all_multi_iso_concrete_clip_continuous.png",
):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_losses, label="Train loss (MSE)")
    plt.plot(epochs, val_losses, label="Val loss (MSE)")
    plt.plot(epochs, pde_losses, label="PDE RGB MSE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / MSE")
    plt.title("Training / Validation / PDE Loss Curves (continuous ablation + CLIP)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# -------------------------------------------------------------------------
# 5. Main: multi-family training + evaluation with CLIP (continuous ablation)
# -------------------------------------------------------------------------

def main():
    family_roots = {
        "lin_iso":   "/data/IMcali/sim_results_2025-12-17 22:31:04_lin_iso",
        "exp_iso":   "/data/IMcali/sim_results_2025-12-19 18:25:27_exp_iso",
        "concrete":  "/data/IMcali/sim_results_2025-12-19 15:52:24_concrete_iso",
    }

    family_text = {
        "exp_iso":  "exponential isotropic diffusion over four equal time intervals",
        "lin_iso":  "linear isotropic diffusion over four equal time intervals",
        "concrete": "chloride diffusion into concrete over four equal time intervals",
    }

    img_size = 128
    patch_size = 2
    batch_size = 16
    num_epochs = 100

    theta_min = GLOBAL_THETA_MIN
    theta_max = GLOBAL_THETA_MAX

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    dataset = MultiFamilyCalibDiffDataset(
        family_roots=family_roots,
        img_size=img_size,
    )

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

    lang_dim = next(iter(fam2lang.values())).shape[-1]

    model = CalibDiffContinuous(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=128,
        num_heads=4,
        num_latents=2048,
        num_latent_layers=6,
        theta_min=theta_min,
        theta_max=theta_max,
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
            f"train loss: {train_loss:.6f} | val loss: {val_loss:.6f} | "
            f"PDE RGB MSE: {pde_rgb_mse:.6f} | "
            f"Conc MSE: {conc_mse:.6e} | "
            f"Param recovery: {param_rec*100:.2f}%"
        )

        plot_all_losses(
            train_loss_hist,
            val_loss_hist,
            pde_rgb_mse_hist,
            out_path="loss_curves_all_multi_iso_concrete_clip_continuous.png",
        )

    torch.save(model.state_dict(), "calibdiff_multi_iso_concrete_clip_continuous_checkpoint.pt")
    print("Training complete, model saved to calibdiff_multi_iso_concrete_clip_continuous_checkpoint.pt")

    np.savez(
        "loss_histories_all_multi_iso_concrete_clip_continuous.npz",
        train_loss=train_loss_hist,
        val_loss=val_loss_hist,
        pde_rgb_mse=pde_rgb_mse_hist,
        conc_mse=conc_mse_hist,
        param_recovery=param_rec_hist,
    )


if __name__ == "__main__":
    main()
