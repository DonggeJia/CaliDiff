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
        K: int = 3,
        theta_min: float = 0.1,
        theta_max: float = 10.0,
    ):
        """
        Args:
            root_dir: folder containing sim_00000_..., sim_00001_..., ...
            img_size: image size after resizing (HxW, assumed square)
            max_boundaries: max number of boundary tuples to encode
            K: number of time intervals (3 in your simulation)
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
        
        H = W = img_size
        yy, xx = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing="ij",
        )
        # pixel indices from 0 .. img_size-1 (kept as floats)
        # channel order: [x_index, y_index] for consistency with (x,y)
        self.pixel_idx = torch.stack([xx, yy], dim=0)  # [2,H,W]

    def __len__(self):
        return len(self.index)


    def _build_config_vector(self, params):
        """
        Build the configuration vector g from params.json
        WITHOUT boundary tuples (boundary is fixed in the PDE generator).
        """
        width_mm = params["width_mm"]
        height_mm = params["height_mm"]
        c_boundary = params["c_boundary"]
        c_init = params["c_init"]
        c_other = params["c_other"]
        time_days = params["time_days"]

        # rough typical scales (tune as you like)
        width_scale = 120.0    # mm
        height_scale = 120.0   # mm
        c_scale = 0.2          # concentration (assuming <= 0.2)
        time_scale = 400.0     # days

        g = [
            width_mm / width_scale,
            height_mm / height_scale,
            c_boundary / c_scale,
            c_init / c_scale,
            c_other / c_scale,
            time_days / time_scale,
        ]

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

        width_mm = params["width_mm"]
        height_mm = params["height_mm"]

        # --- Load images ---
        # current phi at step k (state at t_k)
        phi_k_path = os.path.join(sim_dir, f"phi_interval_{k:02d}.png")
        # goal = final phi at step K (t_K)
        phi_goal_path = os.path.join(sim_dir, f"phi_interval_{self.K:02d}.png")

        rgb_k = self.transform(Image.open(phi_k_path).convert("RGB"))       # [3,H,W]
        rgb_goal = self.transform(Image.open(phi_goal_path).convert("RGB")) # [3,H,W]

        H = W = self.img_size

        # --- Build physical coordinate channels (x_mm, y_mm) ---
        # pixel indices 0..H-1, 0..W-1 → physical coords 0..height_mm / 0..width_mm
        yy, xx = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing="ij",
        )
        # map indices to mm; use (H-1),(W-1) so endpoints map exactly to 0 and width/height
        x_mm = xx / max(W - 1, 1) * float(width_mm)
        y_mm = yy / max(H - 1, 1) * float(height_mm)
        phys_coords = torch.stack([x_mm, y_mm], dim=0)  # [2,H,W]

        # pixel index channels (already precomputed in __init__)
        # shape: [2,H,W], values in [0, img_size-1]
        pixel_idx = self.pixel_idx

        # concat along channel dim: [3 + 2 + 2, H, W] = [7,H,W]
        I_k = torch.cat([rgb_k, pixel_idx], dim=0)
        I_goal = torch.cat([rgb_goal, pixel_idx], dim=0)

        # --- Target parameters ---
        a_x = params["coefficients_ax"][k]
        a_y = params["coefficients_ay"][k]
        target = torch.tensor([a_x, a_y], dtype=torch.float32)

        # time index (normalized)
        tau_k = torch.tensor([k / (self.K - 1)], dtype=torch.float32)

        return {
            "I_k": I_k,          # [5,H,W]
            "I_goal": I_goal,    # [5,H,W]
            "g": g,
            "tau_k": tau_k,
            "target_params": target,
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
        config_dim=6,  # matches Dataset config vector length
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
        self.num_config_tokens = num_config_tokens

        # Patch encoders for current and goal (shared)
        self.patch_embed = PatchEmbed(
            in_ch=5, embed_dim=embed_dim, patch_size=patch_size, img_size=img_size
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

        # Configuration encoder
        self.config_encoder = MLPConfigEncoder(
            in_dim=config_dim + 1,  # +1 for time tau_k
            embed_dim=embed_dim,
            n_tokens=num_config_tokens,
        )

        # Time token (optional, separate from g)
        self.time_token_proj = nn.Linear(1, embed_dim)

        # ---- Unified positional embedding for the full sequence z0 ----
        # Sequence layout: [cur_patches (N_p), goal_patches (N_p), cfg_tokens (N_c), time_token (1)]
        total_tokens = 2 * self.N_p + self.num_config_tokens + 1
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

        # Add modality embeddings to patches (no positions yet)
        z_cur = z_cur + self.mod_cur               # [B,N_p,C]
        z_goal = z_goal + self.mod_goal            # [B,N_p,C]

        # Concatenate patch tokens
        z_patches = torch.cat([z_cur, z_goal], dim=1)  # [B, 2N_p, C]

        # ---- Config + time tokens ----
        # append tau_k to g for the MLP config encoder
        g_time = torch.cat([g, tau_k], dim=-1)   # [B, G+1]
        z_cfg = self.config_encoder(g_time)      # [B,N_c,C]

        # Separate time token (project tau_k)
        z_time = self.time_token_proj(tau_k)  # [B, C]
        z_time = z_time.unsqueeze(1)          # [B,1,C]

        # Full input sequence: patches + config tokens + time token
        z0 = torch.cat([z_patches, z_cfg, z_time], dim=1)  # [B, N, C]
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
    curve_path: str = "train_loss_curve_1.png",
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

        # split logits into x & y logits
        logits_y = Q_logits.mean(dim=2)   # [B, H_p]   aggregation over x
        logits_x = Q_logits.mean(dim=1)   # [B, W_p]   aggregation over y

        i_idx, j_idx = model.discretize_2d(target_params)  # [B], [B]

        loss_y = F.cross_entropy(logits_y, i_idx)
        loss_x = F.cross_entropy(logits_x, j_idx)
        loss = loss_x + loss_y


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
    root_dir = "/data/IMcali/sim_results_2025-12-13 22:27:08"  # TODO: set your dataset path
    img_size = 128
    patch_size = 2
    batch_size = 16
    num_epochs = 40
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
        config_dim=6,  # [width,height,c_boundary,c_init,c_other,time] + 4*4 boundary feats
        num_config_tokens=4,
        theta_min=theta_min,
        theta_max=theta_max,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)#, weight_decay=1e-4)

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = eval_epoch(model, val_loader, device)
        print(f"Epoch {epoch:03d} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f}")

    # Optionally save model
    torch.save(model.state_dict(), "calibdiff_polynomial_checkpoint.pt")
    print("Training complete, model saved to calibdiff_polynomial_checkpoint.pt")


if __name__ == "__main__":
    main()
