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
import re

# -------------------------------------------------------------------------
# External diffusion-family modules (your existing single-family scripts)
# -------------------------------------------------------------------------
import cali1_iso1_lin_iso as lin_iso_mod
import cali1_iso1_exp_iso as exp_iso_mod
import cali1_iso1_concrete as concrete_mod


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
# 0. Ablation: replace CLIP with a self-trained simple language encoder
# -------------------------------------------------------------------------

class SimpleTokenizer:
    """
    Minimal tokenizer for small fixed prompts.
    - lowercases
    - extracts alphanumeric "words"
    - builds vocab from provided texts
    """
    PAD = "<pad>"
    UNK = "<unk>"

    def __init__(self, texts: List[str]):
        self.word2id = {self.PAD: 0, self.UNK: 1}
        self.id2word = {0: self.PAD, 1: self.UNK}

        for t in texts:
            for w in self._split(t):
                if w not in self.word2id:
                    idx = len(self.word2id)
                    self.word2id[w] = idx
                    self.id2word[idx] = w

    def __len__(self):
        return len(self.word2id)

    @staticmethod
    def _split(text: str) -> List[str]:
        # keep it deterministic/simple
        return re.findall(r"[a-z0-9]+", text.lower())

    def encode(self, text: str, max_len: int) -> List[int]:
        ws = self._split(text)
        ids = [self.word2id.get(w, self.word2id[self.UNK]) for w in ws]
        ids = ids[:max_len]
        if len(ids) < max_len:
            ids = ids + [self.word2id[self.PAD]] * (max_len - len(ids))
        return ids


class SimpleTextEncoder(nn.Module):
    """
    A tiny trainable text encoder:
      tokens -> Embedding (+ positional) -> TransformerEncoder -> masked mean pool -> projection.

    Output is a sentence embedding [B, out_dim] that is trained end-to-end.
    """
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        max_len: int = 32,
        token_dim: int = 128,
        nhead: int = 4,
        nlayers: int = 2,
        out_dim: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.max_len = max_len
        self.token_dim = token_dim
        self.out_dim = out_dim

        self.token_emb = nn.Embedding(vocab_size, token_dim, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, token_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=nhead,
            dim_feedforward=4 * token_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)

        self.proj = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, out_dim),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: [B, L] with PAD on the right
        returns:   [B, out_dim]
        """
        B, L = token_ids.shape
        if L != self.max_len:
            raise ValueError(f"Expected token length {self.max_len}, got {L}")

        pos = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, L)  # [B,L]
        x = self.token_emb(token_ids) + self.pos_emb(pos)  # [B,L,D]

        # Transformer attention mask: True means "ignore"
        src_key_padding_mask = (token_ids == self.pad_id)  # [B,L]
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # [B,L,D]

        # masked mean pool
        mask = (token_ids != self.pad_id).float()  # [B,L]
        denom = torch.clamp(mask.sum(dim=1, keepdim=True), min=1.0)  # [B,1]
        x_pool = (x * mask.unsqueeze(-1)).sum(dim=1) / denom  # [B,D]

        emb = self.proj(x_pool)  # [B,out_dim]
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)  # stabilize like CLIP
        return emb


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
# 2. CalibDiff model with language conditioning (unchanged interface)
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
        theta_min=GLOBAL_THETA_MIN,      # global range across families (for *scaled* θ)
        theta_max=GLOBAL_THETA_MAX,
        n_bins=8,           # number of classification bins
        lang_dim: int = 512,
    ):
        """
        CalibDiff with an extra language token.

        theta_min/theta_max are the *global scaled* range shared by all families.
        lang_dim: dimension of language embeddings.
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

        # Modality embeddings: current vs goal
        self.mod_cur = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mod_goal = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Time token
        self.time_token_proj = nn.Linear(1, embed_dim)

        # Language projection: language embedding -> model dimension
        if self.use_lang:
            self.lang_proj = nn.Linear(lang_dim, embed_dim)

        # Unified positional embedding:
        #   [cur_patches (N_p), goal_patches (N_p), time_token (1), lang_token (1)]
        total_tokens = 2 * self.N_p + 1 + (1 if self.use_lang else 0)
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

        # Head: logits over bins + per-bin offsets
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 2 * n_bins),
        )

    def forward(self, I_k, I_goal, tau_k, lang_emb):
        """
        Args:
            I_k:      [B,3,H,W]
            I_goal:   [B,3,H,W]
            tau_k:    [B,1] normalized time index
            lang_emb: [B,lang_dim] language embedding
        Returns:
            logits:  [B, n_bins]
            offsets: [B, n_bins]
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

        # language token
        if self.use_lang:
            z_lang = self.lang_proj(lang_emb)  # [B,C]
            z_lang = z_lang.unsqueeze(1)       # [B,1,C]
            z0 = torch.cat([z_patches, z_time, z_lang], dim=1)  # [B,2N_p+2,C]
        else:
            z0 = torch.cat([z_patches, z_time], dim=1)          # [B,2N_p+1,C]

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

        head_out = self.head(z_cur_pool)         # [B,2*n_bins]
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

    # parameter recovery: 10% of family-specific calibration range in RAW space
    theta_range = theta_max_fam - theta_min_fam
    param_tol = 0.1 * theta_range

    param_success = bool(np.all(np.abs(a_hat_array_raw - a_true_array) < param_tol))

    return conc_mse, rgb_mse, param_success


@torch.no_grad()
def eval_epoch_pde_multi(
    model: CalibDiff,
    ds_or_subset,
    device: torch.device,
    fam2tokens: Dict[str, torch.Tensor],
    lang_encoder: nn.Module,
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
    lang_encoder.eval()

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

        a_hat_list_raw = []  # RAW params (family-specific)

        # trainable language embedding for this family
        tok = fam2tokens[fam].unsqueeze(0).to(device)  # [1,L]
        lang_emb_fam = lang_encoder(tok)               # [1,lang_dim]

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
# 4. Training & validation loops (same loss, + self-trained language)
# -------------------------------------------------------------------------

def _build_lang_batch(
    family_batch: List[str],
    fam2tokens: Dict[str, torch.Tensor],
    lang_encoder: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """
    Convert a list of family names into a [B, lang_dim] tensor
    via a trainable language encoder.
    """
    tok_list = [fam2tokens[fam] for fam in family_batch]           # list of [L]
    tok_batch = torch.stack(tok_list, dim=0).to(device)            # [B,L]
    lang_batch = lang_encoder(tok_batch)                           # [B,lang_dim]
    return lang_batch


@torch.no_grad()
def eval_epoch(
    model: CalibDiff,
    dataloader: DataLoader,
    device: torch.device,
    fam2tokens: Dict[str, torch.Tensor],
    lang_encoder: nn.Module,
):
    model.eval()
    lang_encoder.eval()

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

        lang_batch = _build_lang_batch(family_batch, fam2tokens, lang_encoder, device)

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
    fam2tokens: Dict[str, torch.Tensor],
    lang_encoder: nn.Module,
    log_every: int = 50,
    curve_path: str = "train_loss_curve_multi_iso_concrete_langabl.png",
):
    model.train()
    lang_encoder.train()

    total_loss = 0.0
    n_batches = 0

    loss_history = []

    for step, batch in enumerate(dataloader, start=1):
        I_k = batch["I_k"].to(device)
        I_goal = batch["I_goal"].to(device)
        tau_k = batch["tau_k"].to(device)
        target_params = batch["target_params"].to(device)  # [B,1], GLOBAL θ
        family_batch = batch["family"]  # list of strings

        lang_batch = _build_lang_batch(family_batch, fam2tokens, lang_encoder, device)

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
    out_path: str = "loss_curves_all_multi_iso_concrete_langabl.png",
):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Val loss")
    plt.plot(epochs, pde_losses, label="PDE RGB MSE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / MSE")
    plt.title("Training / Validation / PDE Loss Curves (multi-family + lang ablation)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# -------------------------------------------------------------------------
# 5. Main: multi-family / cross-family training + evaluation (lang ablation)
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

    # ----------------------------------------------------------
    # Language ablation: build tokenizer + trainable text encoder
    # ----------------------------------------------------------
    max_len = 32
    tokenizer = SimpleTokenizer(list(family_text.values()))
    pad_id = tokenizer.word2id[SimpleTokenizer.PAD]

    fam2tokens: Dict[str, torch.Tensor] = {}
    for fam, text in family_text.items():
        ids = tokenizer.encode(text, max_len=max_len)
        fam2tokens[fam] = torch.tensor(ids, dtype=torch.long)  # [L] on CPU

    lang_encoder = SimpleTextEncoder(
        vocab_size=len(tokenizer),
        pad_id=pad_id,
        max_len=max_len,
        token_dim=128,
        nhead=4,
        nlayers=2,
        out_dim=512,
        dropout=0.0,
    ).to(device)

    # infer embedding dimension
    lang_dim = lang_encoder.out_dim

    # ----------------------------------------------------------
    # Model
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
        lang_dim=lang_dim,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in lang_encoder.parameters())
    trainable_params = (
        sum(p.numel() for p in model.parameters() if p.requires_grad)
        + sum(p.numel() for p in lang_encoder.parameters() if p.requires_grad)
    )
    print(f"Total parameters (model+lang):     {total_params:,}")
    print(f"Trainable parameters (model+lang): {trainable_params:,}")

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(lang_encoder.parameters()),
        lr=5e-5
    )

    train_loss_hist = []
    val_loss_hist = []
    pde_rgb_mse_hist = []
    conc_mse_hist = []
    param_rec_hist = []

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, fam2tokens, lang_encoder)
        val_loss = eval_epoch(model, val_loader, device, fam2tokens, lang_encoder)

        pde_rgb_mse, conc_mse, param_rec = eval_epoch_pde_multi(
            model,
            val_set,
            device,
            fam2tokens,
            lang_encoder,
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
            out_path="loss_curves_all_multi_iso_concrete_langabl.png",
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "lang_encoder_state_dict": lang_encoder.state_dict(),
            "tokenizer_word2id": tokenizer.word2id,
            "max_len": max_len,
        },
        "calibdiff_multi_iso_concrete_langabl_checkpoint.pt",
    )
    print("Training complete, checkpoint saved to calibdiff_multi_iso_concrete_langabl_checkpoint.pt")

    np.savez(
        "loss_histories_all_multi_iso_concrete_langabl.npz",
        train_loss=train_loss_hist,
        val_loss=val_loss_hist,
        pde_rgb_mse=pde_rgb_mse_hist,
        conc_mse=conc_mse_hist,
        param_recovery=param_rec_hist,
    )


if __name__ == "__main__":
    main()
