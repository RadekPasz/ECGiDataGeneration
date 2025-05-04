import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from scgan_models import SeqBSPHSPGenerator, SeqBSPHSPDiscriminator
import torch.nn.functional as F

# -------------------------------
# Settings
# -------------------------------
latent_dim = 100
seq_len = 150
epochs = 50
batch_size = 2
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# Dataset
# -------------------------------
class PairedSequenceDataset(Dataset):
    def __init__(self, folder, seq_len=150):
        self.files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.npz')])
        self.seq_len = seq_len

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        bsp = torch.tensor(data["bsp"], dtype=torch.float32)  # Already (T, 1, 128, 128)
        hsp = torch.tensor(data["hsp"], dtype=torch.float32)
        return bsp, hsp


# -------------------------------
# Laplacian utility
# -------------------------------
def fast_laplacian(x):
    B, C, T, H, W = x.shape
    lap = torch.zeros_like(x)
    kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                          dtype=torch.float32, device=x.device).unsqueeze(0).unsqueeze(0)
    for t in range(T):
        frame = x[:, :, t]
        lap[:, :, t] = F.conv2d(frame, kernel, padding=1, groups=C)
    return lap

def lagged_temporal_difference_loss(seq, lag=2):
    if seq.shape[2] <= lag:
        return torch.tensor(0.0, device=seq.device)
    return torch.mean(torch.abs(seq[:, :, lag:, :, :] - seq[:, :, :-lag, :, :]))

# -------------------------------
# Training Loop
# -------------------------------
def train_scgan(data_folder):
    dataset = PairedSequenceDataset(data_folder, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    G = SeqBSPHSPGenerator(latent_dim=latent_dim, seq_len=seq_len).to(device)
    D = SeqBSPHSPDiscriminator().to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion_GAN = nn.BCELoss()
    criterion_L1 = nn.L1Loss()
    scaler = GradScaler()

    for epoch in range(epochs):
        print("Entering epoch", epoch)
        for real_bsp, real_hsp in dataloader:
            real_bsp = real_bsp.to(device)
            real_hsp = real_hsp.to(device)
            print("loaded")

            real_bsp = real_bsp.permute(0, 2, 1, 3, 4)  # [B, 1, T, H, W]
            real_hsp = real_hsp.permute(0, 2, 1, 3, 4)  # [B, 1, T, H, W]
            real_seq = torch.cat([real_bsp, real_hsp], dim=1)  # [B, 2, T, H, W]
            print("permuted")

            z = torch.randn(real_bsp.size(0), latent_dim, device=device)

            with autocast(enabled=(device == "cuda")):
                print("about to generate")
                fake_bsp, fake_hsp = G(z, real_seq)
                print("Generated!")

                pred_real = D(real_bsp, real_hsp)
                pred_fake = D(fake_bsp.detach(), fake_hsp.detach())

                valid = torch.ones_like(pred_real) * 0.9
                fake = torch.zeros_like(pred_fake) + 0.1

                loss_D_real = criterion_GAN(pred_real, valid)
                loss_D_fake = criterion_GAN(pred_fake, fake)
                loss_D = (loss_D_real + loss_D_fake) / 2

            opt_D.zero_grad()
            scaler.scale(loss_D).backward()
            scaler.step(opt_D)
            scaler.update()

            with autocast(enabled=(device == "cuda")):
                loss_Temporal = lagged_temporal_difference_loss(fake_bsp, lag=2) + \
                                lagged_temporal_difference_loss(fake_hsp, lag=2)

                pred_fake = D(fake_bsp, fake_hsp)
                valid = torch.ones_like(pred_fake) * 0.9
                loss_GAN = criterion_GAN(pred_fake, valid)
                loss_L1 = criterion_L1(fake_bsp, real_bsp) + criterion_L1(fake_hsp, real_hsp)
                loss_Lap = torch.mean(torch.abs(fast_laplacian(fake_bsp))) + \
                           torch.mean(torch.abs(fast_laplacian(fake_hsp)))

                #loss_G = loss_GAN + 10 * loss_L1 + 1 * loss_Lap + 1 * loss_Temporal
                loss_G = loss_GAN + 10 * loss_L1 + 1 * loss_Temporal

            opt_G.zero_grad()
            scaler.scale(loss_G).backward()
            scaler.step(opt_G)
            scaler.update()

        print(f"[Epoch {epoch+1}/{epochs}] Loss_G: {loss_G.detach().item():.4f} | Loss_D: {loss_D.detach().item():.4f}")

        #if (epoch + 1) % 10 == 0:
        #    torch.save(G.state_dict(), f"scgan_epoch_{epoch+1}.pth")

    torch.save(G.state_dict(), "scgan_generator_laplacian.pth")
    print("Model saved to scgan_generator_laplacian.pth")

if __name__ == "__main__":
    train_scgan(data_folder="resized_sequence_dataset")
