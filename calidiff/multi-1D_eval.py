"""
Evaluate a trained CalibDiff (multi-family + CLIP) model on each diffusion family
separately (lin_iso / exp_iso / concrete), using 50 trajectories per family.

- Loads checkpoint: calibdiff_multi_iso_concrete_clip_checkpoint.pt
- Builds the same MultiFamilyCalibDiffDataset + CalibDiff architecture
- Builds CLIP text embeddings (fam2lang) exactly like training
- For each family:
    * picks exactly 50 sim_* trajectories (deterministic or random)
    * runs physics-based evaluation (PDE re-simulation) and reports:
        - mean RGB MSE
        - mean concentration-field MSE
        - parameter recovery rate

Run:
  python eval_separate_families.py
"""

import os
import json
import glob
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset

import matplotlib as mpl
from concurrent.futures import ProcessPoolExecutor

# -------------------------------------------------------------------------
# External diffusion-family modules (must match your training environment)
# -------------------------------------------------------------------------
import cali1_iso1_lin_iso as lin_iso_mod
import cali1_iso1_exp_iso as exp_iso_mod
import cali1_iso1_concrete as concrete_mod

# -------------------------------------------------------------------------
# CLIP
# -------------------------------------------------------------------------
import clip


# =========================
# Colormap helper
# =========================
_base_cmap = mpl.colormaps.get_cmap("nipy_spectral")
_highres_cmap = _base_cmap.resampled(1024)


def phi_to_rgb_colormap(phi_np, vmin=0.0, vmax=0.2):
    """
    Map scalar field φ to RGB using the same colormap and vmin/vmax
    as in the data generator.
    phi_np: [ny, nx] numpy array
    returns: [ny, nx, 3] float32 in [0,1]
    """
    phi_clipped = np.clip(phi_np, vmin, vmax)
    norm = (phi_clipped - vmin) / (vmax - vmin + 1e-8)
    rgba = _highres_cmap(norm)  # [ny, nx, 4]
    rgb = rgba[..., :3]
    return rgb.astype(np.float32)


# =========================
# Parameter scaling
# =========================
FAMILY_THETA: Dict[str, Tuple[float, float]] = {
    "lin_iso": (1.0, 10.0),     # a(t) in [1, 10]
    "exp_iso": (1.0, 10.0),     # a(t) in [1, 10]
    "concrete": (0.1, 1.0),     # A0(t) in [0.1, 1.0]
}

GLOBAL_THETA_MIN = 1.0
GLOBAL_THETA_MAX = 10.0


def scale_param_to_global(family: str, a_raw: float) -> float:
    theta_min_fam, theta_max_fam = FAMILY_THETA[family]
    if theta_max_fam <= theta_min_fam:
        return GLOBAL_THETA_MIN
    t = (a_raw - theta_min_fam) / (theta_max_fam - theta_min_fam)
    return GLOBAL_THETA_MIN + t * (GLOBAL_THETA_MAX - GLOBAL_THETA_MIN)


def unscale_param_from_global(family: str, a_global: float) -> float:
    theta_min_fam, theta_max_fam = FAMILY_THETA[family]
    if GLOBAL_THETA_MAX <= GLOBAL_THETA_MIN:
        return theta_min_fam
    t = (a_global - GLOBAL_THETA_MIN) / (GLOBAL_THETA_MAX - GLOBAL_THETA_MIN)
    return theta_min_fam + t * (theta_max_fam - theta_min_fam)


# =========================
# Dataset
# =========================
class MultiFamilyCalibDiffDataset(Dataset):
    """
    Mixed dataset over multiple diffusion 'families', all with a single scalar
    parameter per time interval stored in params["coefficients_ax"].

    Dataset returns target_params in the GLOBAL scaled θ-space.
    """

    def __init__(self, family_roots: Dict[str, str], img_size: int = 128):
        super().__init__()
        self.family_roots = dict(family_roots)
        self.img_size = img_size

        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
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

        self.index = self.items  # mirrors your original code

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
        target = torch.tensor([a_k_global], dtype=torch.float32)

        phi_k_path = os.path.join(sim_dir, f"phi_interval_{k:02d}.png")
        phi_goal_path = os.path.join(sim_dir, f"phi_interval_{K:02d}.png")

        I_k = self.transform(Image.open(phi_k_path).convert("RGB"))
        I_goal = self.transform(Image.open(phi_goal_path).convert("RGB"))
        tau_k = torch.tensor([k / (K - 1)], dtype=torch.float32)

        return {
            "I_k": I_k,
            "I_goal": I_goal,
            "tau_k": tau_k,
            "target_params": target,   # global θ
            "family": fam,
            "sim_idx": sim_idx,
            "k": k,
        }


# =========================
# Model (must match training)
# =========================
class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, embed_dim=128, patch_size=8, img_size=128):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.conv(x)  # [B,C,H_p,W_p]
        B, C, H_p, W_p = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N_p, C]
        return x, H_p, W_p


class LatentBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
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
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_ctx = nn.LayerNorm(dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor):
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
        theta_min=GLOBAL_THETA_MIN,
        theta_max=GLOBAL_THETA_MAX,
        n_bins=8,
        lang_dim: int = 512,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.n_bins = n_bins
        self.lang_dim = lang_dim
        self.use_lang = lang_dim is not None and lang_dim > 0

        self.patch_embed = PatchEmbed(in_ch=3, embed_dim=embed_dim, patch_size=patch_size, img_size=img_size)

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
        self.latent_blocks = nn.ModuleList([LatentBlock(embed_dim, num_heads) for _ in range(num_latent_layers)])
        self.cross_attn_out = CrossAttention(embed_dim, num_heads)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 2 * n_bins),
        )

    def forward(self, I_k, I_goal, tau_k, lang_emb):
        B = I_k.shape[0]

        z_cur, H_p, W_p = self.patch_embed(I_k)
        z_goal, _, _ = self.patch_embed(I_goal)
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
        zout = self.cross_attn_out(z0, latents)

        z_cur_out = zout[:, : self.N_p, :]
        z_cur_pool = z_cur_out.mean(dim=1)

        head_out = self.head(z_cur_pool)
        logits, offset_raw = head_out.chunk(2, dim=-1)
        offsets = 0.5 * torch.tanh(offset_raw)
        return logits, offsets

    def discretize_1d(self, params: torch.Tensor):
        a = params[:, 0]
        v_clamp = torch.clamp(a, self.theta_min, self.theta_max)
        ratio = (v_clamp - self.theta_min) / max(self.theta_max - self.theta_min, 1e-6)
        idx = torch.round(ratio * (self.n_bins - 1)).long()
        idx = torch.clamp(idx, 0, self.n_bins - 1)
        return idx

    def bins_to_param(self, bin_idx: torch.Tensor, offsets: torch.Tensor):
        bin_idx_long = bin_idx.long()
        B = bin_idx_long.shape[0]
        offset_sel = offsets.gather(1, bin_idx_long.view(B, 1)).squeeze(1)
        bin_idx_f = bin_idx_long.float()
        bin_width = (self.theta_max - self.theta_min) / self.n_bins
        bin_centers = self.theta_min + (bin_idx_f + 0.5) / self.n_bins * (self.theta_max - self.theta_min)
        a_hat = bin_centers + offset_sel * bin_width
        return a_hat


# =========================
# PDE worker + evaluation
# =========================
def _multi_pde_worker_task(args):
    (
        family,
        sim_dir,
        params,
        a_hat_array_raw,  # RAW family-specific range
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
        phi_hat = lin_iso_mod.simulate_final_phi_isotropic(params, a_hat_array_raw, nx=nx_sim, ny=ny_sim)
        phi_true = lin_iso_mod.simulate_final_phi_isotropic(params, a_true_array, nx=nx_sim, ny=ny_sim)
    elif family == "exp_iso":
        phi_hat = exp_iso_mod.simulate_final_phi_isotropic(params, a_hat_array_raw, nx=nx_sim, ny=ny_sim)
        phi_true = exp_iso_mod.simulate_final_phi_isotropic(params, a_true_array, nx=nx_sim, ny=ny_sim)
    elif family == "concrete":
        phi_hat = concrete_mod.simulate_final_phi_chloride(params, a_hat_array_raw, nx=nx_sim, ny=ny_sim)
        phi_true = concrete_mod.simulate_final_phi_chloride(params, a_true_array, nx=nx_sim, ny=ny_sim)
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
    model: CalibDiff,
    ds_or_subset,
    device: torch.device,
    fam2lang: Dict[str, torch.Tensor],
    max_sims: Optional[int] = 50,  # number of trajectories to use; None => use all
    nx_sim: int = 64,
    ny_sim: int = 64,
    vmin: float = 0.0,
    vmax: float = 0.2,
    num_workers: int = 14,
):
    model.eval()

    if isinstance(ds_or_subset, Subset):
        base_ds = ds_or_subset.dataset
        idx_list = ds_or_subset.indices
    else:
        base_ds = ds_or_subset
        idx_list = list(range(len(base_ds)))

    assert isinstance(base_ds, MultiFamilyCalibDiffDataset)

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

            I_k = base_ds.transform(Image.open(phi_k_path).convert("RGB")).unsqueeze(0).to(device)
            I_goal = base_ds.transform(Image.open(phi_goal_path).convert("RGB")).unsqueeze(0).to(device)
            tau_k = torch.tensor([[k / (K - 1)]], dtype=torch.float32).to(device)

            logits, offsets = model(I_k, I_goal, tau_k, lang_emb_fam)
            pred_bins = torch.argmax(logits, dim=-1)

            a_hat_global = model.bins_to_param(pred_bins, offsets)[0].item()
            a_hat_raw = unscale_param_from_global(fam, a_hat_global)
            a_hat_list_raw.append(a_hat_raw)

        a_hat_array_raw = np.array(a_hat_list_raw, dtype=float)

        theta_min_fam, theta_max_fam = FAMILY_THETA[fam]
        tasks.append((
            fam, sim_dir, params, a_hat_array_raw, K, nx_sim, ny_sim,
            img_size, vmin, vmax, theta_min_fam, theta_max_fam
        ))

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
    return mean_rgb_mse, mean_conc_mse, param_recovery_rate, n_sims_used


# =========================
# Family subset helper
# =========================
def make_family_subset_exact_sims(
    dataset: MultiFamilyCalibDiffDataset,
    family: str,
    n_sims: int = 50,
    deterministic_first: bool = True,
    seed: int = 0,
) -> Tuple[Subset, List[int]]:
    """
    Build a Subset that contains *all k* samples for exactly n_sims trajectories
    in a given family.

    Returns:
      subset, chosen_sim_indices
    """
    all_sim_indices = list(range(len(dataset.family_info[family]["sim_dirs"])))
    if len(all_sim_indices) < n_sims:
        raise ValueError(f"Family '{family}' has only {len(all_sim_indices)} sims (< {n_sims}).")

    if deterministic_first:
        chosen = all_sim_indices[:n_sims]
    else:
        rng = random.Random(seed)
        chosen = rng.sample(all_sim_indices, n_sims)

    chosen_set = set(chosen)
    subset_indices = [
        i for i, (fam, sim_idx, k) in enumerate(dataset.index)
        if fam == family and sim_idx in chosen_set
    ]
    return Subset(dataset, subset_indices), chosen


# =========================
# Main evaluation script
# =========================
def main():
    # ---- Match your training paths ----
    family_roots = {
        "lin_iso":   "/data/IMcali/sim_results_2025-12-17 22:31:04_lin_iso",
        "exp_iso":   "/data/IMcali/sim_results_2025-12-19 18:25:27_exp_iso",
        "concrete":  "/data/IMcali/sim_results_2025-12-19 15:52:24_concrete_iso",
    }

    # Must match training text prompts (or at least keys)
    family_text = {
        "exp_iso":  "exponential isotropic diffusion over four equal time intervals",
        "lin_iso":  "linear isotropic diffusion over four equal time intervals",
        "concrete": "chloride diffusion into concrete over four equal time intervals",
    }

    # ---- Must match training architecture ----
    checkpoint_path = "calibdiff_multi_iso_concrete_clip_checkpoint.pt"
    img_size = 128
    patch_size = 2
    embed_dim = 128
    num_heads = 4
    num_latents = 2048
    num_latent_layers = 6
    n_bins = 8
    theta_min = GLOBAL_THETA_MIN
    theta_max = GLOBAL_THETA_MAX

    # ---- Evaluation settings ----
    sims_per_family = 50
    deterministic_first_50 = True   # set False to random-sample 50 per family
    selection_seed = 0              # used only if deterministic_first_50=False

    max_sims_inside_eval = None     # IMPORTANT: None => use exactly the sims in the subset
    nx_sim = 64
    ny_sim = 64
    num_workers = 14
    vmin, vmax = 0.0, 0.2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")

    # ---- Build dataset (only used for loading images + listing sims) ----
    dataset = MultiFamilyCalibDiffDataset(family_roots=family_roots, img_size=img_size)

    # ---- Build CLIP language embeddings (same as training) ----
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    fam2lang: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for fam, text in family_text.items():
            tokens = clip.tokenize([text]).to(device)
            sentence_emb = clip_model.encode_text(tokens)
            sentence_emb = sentence_emb / sentence_emb.norm(dim=-1, keepdim=True)
            sentence_emb = sentence_emb.to(dtype=torch.float32)
            fam2lang[fam] = sentence_emb.squeeze(0).detach().cpu()

    lang_dim = next(iter(fam2lang.values())).shape[-1]

    # ---- Build model + load weights ----
    model = CalibDiff(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_latents=num_latents,
        num_latent_layers=num_latent_layers,
        theta_min=theta_min,
        theta_max=theta_max,
        n_bins=n_bins,
        lang_dim=lang_dim,
    ).to(device)

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=True)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  missing keys: {len(missing)} | unexpected keys: {len(unexpected)}")

    # ---- Evaluate each family separately ----
    print("\n================= Per-family PDE evaluation =================")
    results = {}

    for fam in ["lin_iso", "exp_iso", "concrete"]:
        subset, chosen_sims = make_family_subset_exact_sims(
            dataset,
            family=fam,
            n_sims=sims_per_family,
            deterministic_first=deterministic_first_50,
            seed=selection_seed,
        )

        mean_rgb_mse, mean_conc_mse, param_rec, used = eval_epoch_pde_multi(
            model=model,
            ds_or_subset=subset,
            device=device,
            fam2lang=fam2lang,
            max_sims=max_sims_inside_eval,  # None => use exactly chosen sims
            nx_sim=nx_sim,
            ny_sim=ny_sim,
            vmin=vmin,
            vmax=vmax,
            num_workers=num_workers,
        )

        results[fam] = {
            "n_sims": used,
            "mean_rgb_mse": float(mean_rgb_mse),
            "mean_conc_mse": float(mean_conc_mse),
            "param_recovery_rate": float(param_rec),
            "chosen_sim_indices": chosen_sims,
        }

        print(
            f"\nFamily: {fam}\n"
            f"  sims used:       {used}\n"
            f"  mean RGB MSE:    {mean_rgb_mse:.6f}\n"
            f"  mean Conc MSE:   {mean_conc_mse:.6e}\n"
            f"  param recovery:  {param_rec*100:.2f}%"
        )

    # Optional: save results JSON
    out_json = "per_family_eval_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved per-family results to: {out_json}")


if __name__ == "__main__":
    main()
