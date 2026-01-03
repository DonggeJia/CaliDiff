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


# ============================================================
# 1. Dataset: Expert Trajectories from sim_* folders
# ============================================================

class CalibDiffDataset(Dataset):
    """
    Each item is one (trajectory, time step) pair:
        inputs:  I_k (current), I_goal (final), config g, time index k
        target:  bin index for (a_x[k], a_y[k]) as a 2D bin (i, j)
    """

    def __init__(
        self,
        root_dir: str,
        img_size: int = 128,
        max_boundaries: int = 4,
        K: int = 10,
        theta_min: float = 0.1,
        theta_max: float = 10.0,
    ):
        """
        Args:
            root_dir: folder containing sim_00000_..., sim_00001_..., ...
            img_size: image size after resizing (HxW, assumed square)
            max_boundaries: max number of boundary tuples to encode
            K: number of time intervals (10 in your simulation)
            theta_min, theta_max: global range for coefficients_ax/ay
                                  for binning (can be tuned based on data stats)
        """
        self.root_dir = root_dir
        self.img_size = img_size
        self.max_boundaries = max_boundaries
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
            for k in range(self.K):  # 0..9
                self.index.append((sim_idx, k))

    def __len__(self):
        return len(self.index)

    def _encode_boundaries(self, boundaries_tuples, width_mm, height_mm):
        """
        Encode up to max_boundaries tuples as normalized features.
        boundaries_tuples is a list of pairs of [start, end] indices, e.g.
          [
            [[0, 74], [0, 74]],
            ...
          ]
        We flatten each into 4 numbers and pad with zeros if fewer than max_boundaries.
        """
        # Flatten and normalize
        encoded = []
        for bt in boundaries_tuples[: self.max_boundaries]:
            # bt is ((h1_mm, w1_mm), (h2_mm, w2_mm))
            (h1_mm, w1_mm), (h2_mm, w2_mm) = bt

            encoded.extend([
                h1_mm / max(height_mm, 1e-6),
                h2_mm / max(height_mm, 1e-6),
                w1_mm / max(width_mm, 1e-6),
                w2_mm / max(width_mm, 1e-6),
            ])
        # Pad to fixed length
        needed = self.max_boundaries * 4 - len(encoded)
        if needed > 0:
            encoded.extend([0.0] * needed)
        return encoded

    def _build_config_vector(self, params):
        """
        Build the configuration vector g from params.json
        """
        width_mm = params["width_mm"]
        height_mm = params["height_mm"]
        boundaries_tuples = params["boundaries_tuples"]
        c_boundary = params["c_boundary"]
        c_init = params["c_init"]
        c_other = params["c_other"]
        time_days = params["time_days"]

        # Encode boundaries
        boundary_feats = self._encode_boundaries(
            boundaries_tuples, width_mm, height_mm
        )

        # rough typical scales (tune as you like)
        width_scale = 120.0    # mm
        height_scale = 120.0   # mm
        c_scale = 0.2          # concentration (assuming <= 0.1)
        time_scale = 400.0     # days

        g = [
            width_mm / width_scale,
            height_mm / height_scale,
            c_boundary / c_scale,
            c_init / c_scale,
            c_other / c_scale,
            time_days / time_scale,
        ] + boundary_feats   # boundary_feats already normalized

        g = torch.tensor(g, dtype=torch.float32)
        return g

    def _discretize_param(self, value: float, n_bins: int) -> int:
        """
        Map continuous parameter value to integer bin index in [0, n_bins-1].
        """
        v = max(min(value, self.theta_max), self.theta_min)
        ratio = (v - self.theta_min) / max(self.theta_max - self.theta_min, 1e-6)
        idx = int(round(ratio * (n_bins - 1)))
        idx = max(0, min(idx, n_bins - 1))
        return idx

    def __getitem__(self, idx):
        sim_idx, k = self.index[idx]
        sim_dir = self.sim_dirs[sim_idx]

        # --- Load params.json ---
        with open(os.path.join(sim_dir, "params.json"), "r") as f:
            params = json.load(f)

        # Configuration vector (no time here; time index handled in model input)
        g = self._build_config_vector(params)  # [G]

        # --- Load images ---
        # current phi at step k
        # current phi at step k (state at t_k, *before* applying interval k)
        phi_k_path = os.path.join(sim_dir, f"phi_interval_{k:02d}.png")

        # goal = final phi at step K (t_{K}), not K-1
        phi_goal_path = os.path.join(sim_dir, f"phi_interval_{self.K:02d}.png")


        I_k = self.transform(Image.open(phi_k_path).convert("RGB"))     # [3,H,W]
        I_goal = self.transform(Image.open(phi_goal_path).convert("RGB"))  # [3,H,W]

        # --- Target parameters ---
        a_x = params["coefficients_ax"][k]
        a_y = params["coefficients_ay"][k]

        # bin indices for x/y
        # We will later define N_bins_x=W_p, N_bins_y=H_p, but for the dataset we
        # only store raw float values; binning can be done inside the model or collate.
        target = torch.tensor([a_x, a_y], dtype=torch.float32)

        # time index (normalized)
        tau_k = torch.tensor([k / (self.K - 1)], dtype=torch.float32)

        return {
            "I_k": I_k,
            "I_goal": I_goal,
            "g": g,
            "tau_k": tau_k,
            "target_params": target,  # continuous; discretized in training step
        }


# ============================================================
# 2. CalibDiff model: patch embedding + latent Transformer + 2D Q-map
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
        # H_p = W_p = img_size / patch_size (assumed divisible)

    def forward(self, x):
        # x: [B,3,H,W] -> [B,C,H_p,W_p]
        x = self.conv(x)
        B, C, H_p, W_p = x.shape
        # flatten to tokens
        x = x.flatten(2).transpose(1, 2)  # [B, N_p, C]
        return x, H_p, W_p


class MLPConfigEncoder(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int, n_tokens: int = 4):
        super().__init__()
        self.n_tokens = n_tokens
        self.fc = nn.Sequential(
            nn.Linear(in_dim, embed_dim * n_tokens),
            nn.ReLU(),
            nn.Linear(embed_dim * n_tokens, embed_dim * n_tokens),
            nn.ReLU(),
        )
        self.embed_dim = embed_dim

    def forward(self, g: torch.Tensor):
        # g: [B, G]
        B = g.shape[0]
        x = self.fc(g)  # [B, embed_dim * n_tokens]
        x = x.view(B, self.n_tokens, self.embed_dim)  # [B, N_c, C]
        return x


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
        config_dim=6 + 4 * 4,  # matches Dataset config vector length
        num_config_tokens=4,
        theta_min=0.1,
        theta_max=10.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.theta_min = theta_min
        self.theta_max = theta_max

        # Patch encoders for current and goal
        self.patch_embed = PatchEmbed(
            in_ch=3, embed_dim=embed_dim, patch_size=patch_size, img_size=img_size
        )  # shared for cur/goal for simplicity

        # Positional encodings for patches (2D -> 1D tokens)
        # We'll create learnable [N_p, C] and broadcast across batch
        H_p = img_size // patch_size
        W_p = img_size // patch_size
        self.H_p = H_p
        self.W_p = W_p
        self.N_p = H_p * W_p
        self.pos_embed_patches = nn.Parameter(
            torch.randn(2 * self.N_p, embed_dim) * 0.01
        )  # for cur+goal tokens

        # Modality embeddings: current vs goal
        self.mod_cur = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mod_goal = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Configuration encoder
        self.config_encoder = MLPConfigEncoder(
            in_dim=config_dim + 1,  # +1 for time tau_k
            embed_dim=embed_dim,
            n_tokens=num_config_tokens,
        )

        # Time token (optional, but we already put tau into config; can keep this simple)
        self.time_token_proj = nn.Linear(1, embed_dim)

        # Latent bottleneck
        self.latents = nn.Parameter(torch.randn(1, num_latents, embed_dim) * 0.02)
        self.cross_attn_in = CrossAttention(embed_dim, num_heads)
        self.latent_blocks = nn.ModuleList(
            [LatentBlock(embed_dim, num_heads) for _ in range(num_latent_layers)]
        )
        self.cross_attn_out = CrossAttention(embed_dim, num_heads)

        # Q-map decoder: from current-image tokens to [H_p,W_p] heatmap
        self.dec_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, 1, kernel_size=1),
        )

    def forward(self, I_k, I_goal, g, tau_k):
        """
        Args:
            I_k:    [B,3,H,W]
            I_goal: [B,3,H,W]
            g:      [B,G]
            tau_k:  [B,1] normalized time index
        Returns:
            Q_logits: [B, H_p, W_p] unnormalized logits over 2D bins
        """
        B = I_k.shape[0]

        # ---- Patch embedding for current and goal ----
        z_cur, H_p, W_p = self.patch_embed(I_k)     # [B,N_p,C]
        z_goal, _, _ = self.patch_embed(I_goal)     # [B,N_p,C]

        # sanity check
        assert H_p == self.H_p and W_p == self.W_p

        # positional + modality embeddings
        # concatenate cur and goal before adding pos
        z = torch.cat([z_cur, z_goal], dim=1)  # [B, 2N_p, C]
        pos = self.pos_embed_patches.unsqueeze(0).expand(B, -1, -1)  # [B,2N_p,C]
        z = z + pos

        # Add modality embeddings
        z_cur = z[:, : self.N_p, :] + self.mod_cur
        z_goal = z[:, self.N_p :, :] + self.mod_goal
        z_patches = torch.cat([z_cur, z_goal], dim=1)  # [B,2N_p,C]

        # ---- Config + time tokens ----
        # append tau_k to g
        g_time = torch.cat([g, tau_k], dim=-1)  # [B, G+1]
        z_cfg = self.config_encoder(g_time)      # [B,N_c,C]
        N_c = z_cfg.shape[1]

        # Separate time token (optional)
        z_time = self.time_token_proj(tau_k)  # [B, C]
        z_time = z_time.unsqueeze(1)          # [B,1,C]

        # Full input sequence
        z0 = torch.cat([z_patches, z_cfg, z_time], dim=1)  # [B, N, C]
        N = z0.shape[1]

        # ---- Latent bottleneck ----
        latents = self.latents.expand(B, -1, -1)  # [B,L,C]

        # cross-attn in
        latents = self.cross_attn_in(latents, z0)
        # latent self-attention blocks
        for blk in self.latent_blocks:
            latents = blk(latents)
        # cross-attn out (tokens query, latents as context)
        zout = self.cross_attn_out(z0, latents)  # [B,N,C]

        # ---- Q-map decoding from current-image tokens ----
        # current tokens are first N_p entries
        z_cur_out = zout[:, : self.N_p, :]  # [B,N_p,C]
        # reshape to [B,C,H_p,W_p]
        z_cur_out = z_cur_out.transpose(1, 2).view(B, self.embed_dim, H_p, W_p)
        # conv head
        Q = self.dec_conv(z_cur_out).squeeze(1)  # [B,H_p,W_p]
        return Q  # logits

    def discretize_2d(self, params: torch.Tensor):
        """
        Convert continuous params [B,2] into (i,j) bin indices
        using [theta_min, theta_max] mapped onto H_p, W_p.
        """
        a_x = params[:, 0]
        a_y = params[:, 1]

        def one_dim(v, n_bins):
            v_clamp = torch.clamp(v, self.theta_min, self.theta_max)
            ratio = (v_clamp - self.theta_min) / max(self.theta_max - self.theta_min, 1e-6)
            idx = torch.round(ratio * (n_bins - 1)).long()
            idx = torch.clamp(idx, 0, n_bins - 1)
            return idx

        j = one_dim(a_x, self.W_p)  # horizontal bins
        i = one_dim(a_y, self.H_p)  # vertical bins
        return i, j

    def bins_to_params(self, i: torch.Tensor, j: torch.Tensor):
        """
        Map bin indices (i,j) back to continuous param estimates.
        """
        device = i.device
        i = i.float()
        j = j.float()
        # centers of bins
        a_x = self.theta_min + (j + 0.5) / self.W_p * (self.theta_max - self.theta_min)
        a_y = self.theta_min + (i + 0.5) / self.H_p * (self.theta_max - self.theta_min)
        return torch.stack([a_x, a_y], dim=-1).to(device)


# ============================================================
# 3. Training Loop
# ============================================================
def train_epoch(
    model: CalibDiff,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_every: int = 50,
    curve_path: str = "train_loss_curve.png",
):
    model.train()
    total_loss = 0.0
    n_batches = 0

    loss_history = []  # store per-batch loss within this epoch

    for step, batch in enumerate(dataloader, start=1):
        I_k = batch["I_k"].to(device)           # [B,3,H,W]
        I_goal = batch["I_goal"].to(device)
        g = batch["g"].to(device)               # [B,G]
        tau_k = batch["tau_k"].to(device)       # [B,1]
        target_params = batch["target_params"].to(device)  # [B,2]

        optimizer.zero_grad()

        # forward
        Q_logits = model(I_k, I_goal, g, tau_k)  # [B,H_p,W_p]
        B, H_p, W_p = Q_logits.shape

        # discretize ground-truth params to (i,j) bins
        i_idx, j_idx = model.discretize_2d(target_params)  # [B], [B]

        # joint index for cross-entropy
        target_flat = i_idx * W_p + j_idx  # [B]

        # flatten logits
        Q_flat = Q_logits.view(B, H_p * W_p)  # [B,H_p*W_p]

        loss = F.cross_entropy(Q_flat, target_flat)
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

            # Overwrite the previous image every `log_every` steps
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
        g = batch["g"].to(device)
        tau_k = batch["tau_k"].to(device)
        target_params = batch["target_params"].to(device)

        Q_logits = model(I_k, I_goal, g, tau_k)
        B, H_p, W_p = Q_logits.shape

        i_idx, j_idx = model.discretize_2d(target_params)
        target_flat = i_idx * W_p + j_idx
        Q_flat = Q_logits.view(B, H_p * W_p)

        loss = F.cross_entropy(Q_flat, target_flat)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ============================================================
# 4. Main entry point
# ============================================================

def main():
    root_dir = "/data/IMcali/sim_results_2025-12-12"  # TODO: set your dataset path
    img_size = 128
    patch_size = 2
    batch_size = 16
    num_epochs = 20
    theta_min = 0.1    # tune based on your coefficient ranges
    theta_max = 10.0

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
    # train_loader = DataLoader(
    #     train_set,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=0,      # <--- IMPORTANT
    #     pin_memory=False,   # safer to start with
    # )

    # val_loader = DataLoader(
    #     val_set,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=0,      # <--- IMPORTANT
    #     pin_memory=False,
    # )


    # Model
    model = CalibDiff(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=128,
        num_heads=4,
        num_latents=2048,
        num_latent_layers=6,
        config_dim=6 + 4 * 4,  # [width,height,c_boundary,c_init,c_other,time] + 4*4 boundary feats
        num_config_tokens=4,
        theta_min=theta_min,
        theta_max=theta_max,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = eval_epoch(model, val_loader, device)
        print(f"Epoch {epoch:03d} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f}")

    # Optionally save model
    torch.save(model.state_dict(), "calibdiff_polynomial_checkpoint.pt")
    print("Training complete, model saved to calibdiff_polynomial_checkpoint.pt")


if __name__ == "__main__":
    main()
