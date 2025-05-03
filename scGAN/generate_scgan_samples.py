import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scgan_models import SeqBSPHSPGenerator
import imageio
import torch.nn.functional as F

# Settings
latent_dim = 100
num_samples = 152
#seq_len = 150 
frames_to_save = 20

# Output directories
gif_dir = "generated_gifs"
img_dir = "generated_samples"
npz_dir = "generated_npz"
bspm_path = os.path.join(img_dir, "BSPM")
hspm_path = os.path.join(img_dir, "HSPM")

# Create directories
os.makedirs(gif_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)
os.makedirs(npz_dir, exist_ok=True)
os.makedirs(bspm_path, exist_ok=True)
os.makedirs(hspm_path, exist_ok=True)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = SeqBSPHSPGenerator(latent_dim=latent_dim, seq_len=150).to(device)
G.load_state_dict(torch.load("scgan_generator_laplacian.pth", map_location=device))
G.eval()

# Load some real samples for conditioning
real_dataset = "sequence_dataset"
real_files = sorted([f for f in os.listdir(real_dataset) if f.endswith(".npz")])

# Generate samples
with torch.no_grad():
    for i in range(num_samples):
        # Load a random real sample for conditioning
        real_path = os.path.join(real_dataset, np.random.choice(real_files))
        real_data = np.load(real_path)

        bsp_real = real_data["bsp"]  
        hsp_real = real_data["hsp"]

        # Prepare condition tensor
        bsp_real = torch.tensor(bsp_real, dtype=torch.float32, device=device)  # (1, 1, T, H, W)
        hsp_real = torch.tensor(hsp_real, dtype=torch.float32, device=device)

        if bsp_real.ndim == 3:  # (T, H, W)
            bsp_real = bsp_real.unsqueeze(0).unsqueeze(0)  # (1, 1, T, H, W)
        elif bsp_real.ndim == 4:  # (C, T, H, W) or (T, C, H, W)
            bsp_real = bsp_real.unsqueeze(0)  # (1, C, T, H, W) maybe

        if bsp_real.shape[1] != 1:
            bsp_real = bsp_real.permute(0, 2, 1, 3, 4)  # (1, 1, T, H, W)

        if hsp_real.ndim == 3:
            hsp_real = hsp_real.unsqueeze(0).unsqueeze(0)
        elif hsp_real.ndim == 4:
            hsp_real = hsp_real.unsqueeze(0)

        if hsp_real.shape[1] != 1:
            hsp_real = hsp_real.permute(0, 2, 1, 3, 4)

        real_condition = torch.cat([bsp_real, hsp_real], dim=1)  # (1, 2, T, H, W)
        real_condition = F.interpolate(real_condition, size=(150, 128, 128), mode="trilinear", align_corners=False)

        # Generate synthetic sequence
        z = torch.randn(1, latent_dim, device=device)
        fake_bsp, fake_hsp = G(z, real_condition)
        seq_len = 150
        print("seq_len", seq_len)

        # Normalize from [-1, 1] to [0, 1]
        fake_bsp = (fake_bsp + 1) / 2
        fake_hsp = (fake_hsp + 1) / 2

        # Save full sequence to .npz
        np.savez_compressed(
            os.path.join(npz_dir, f"sample_{i:02d}.npz"),
            bsp=fake_bsp.cpu().numpy(),
            hsp=fake_hsp.cpu().numpy()
        )

        # Save first 20 frames of BSP and HSP
        T = min(fake_bsp.shape[2], frames_to_save)
        for t in range(T):
            plt.imsave(
                os.path.join(bspm_path, f"generated_bspm_{i:02d}_t{t:02d}.png"),
                fake_bsp[0, 0, t].cpu().numpy(),
                cmap='jet', vmin=0, vmax=1
            )
            plt.imsave(
                os.path.join(hspm_path, f"generated_hspm_{i:02d}_t{t:02d}.png"),
                fake_hsp[0, 0, t].cpu().numpy(),
                cmap='jet', vmin=0, vmax=1
            )

        frames = []
        for t in range(seq_len):
            fig, axes = plt.subplots(1, 2, figsize=(6, 3))
            axes[0].imshow(fake_bsp[0, 0, t].cpu(), cmap='jet', vmin=0, vmax=1)
            axes[0].set_title(f"BSP t={t}")
            axes[0].axis('off')

            axes[1].imshow(fake_hsp[0, 0, t].cpu(), cmap='jet', vmin=0, vmax=1)
            axes[1].set_title(f"HSP t={t}")
            axes[1].axis('off')

            plt.tight_layout()
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.get_renderer().buffer_rgba(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            frames.append(image)
            plt.close()

        gif_path = os.path.join(gif_dir, f"pair_{i:02d}.gif")
        imageio.mimsave(gif_path, frames, fps=2)

print(f"✅ Saved {num_samples} sequences as:\n• PNGs → '{img_dir}/BSPM' & 'HSPM'\n• .npz → '{npz_dir}/'\n• GIFs → '{gif_dir}/'")
