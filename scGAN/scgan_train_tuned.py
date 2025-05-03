
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scgan_models import SeqBSPHSPGenerator, SeqBSPHSPDiscriminator

class PairedSequenceDataset(Dataset):
    def __init__(self, folder, seq_len=15):
        self.files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.npz')])
        self.seq_len = seq_len

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        bsp = data["bsp"]
        hsp = data["hsp"]
        bsp = torch.tensor(bsp, dtype=torch.float32)
        hsp = torch.tensor(hsp, dtype=torch.float32)
        return bsp, hsp

def train_scgan(data_folder, epochs=50, batch_size=2, latent_dim=100, device="cuda"):
    dataset = PairedSequenceDataset(data_folder)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    G = SeqBSPHSPGenerator(latent_dim=latent_dim).to(device)
    D = SeqBSPHSPDiscriminator().to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion_GAN = nn.BCELoss()
    criterion_L1 = nn.L1Loss()

    for epoch in range(epochs):
        for i, (real_bsp, real_hsp) in enumerate(dataloader):
            real_bsp = real_bsp.to(device).permute(0, 2, 1, 3, 4)
            real_hsp = real_hsp.to(device).permute(0, 2, 1, 3, 4)

            z = torch.randn(real_bsp.size(0), latent_dim, device=device)
            fake_bsp, fake_hsp = G(z)

            pred_real = D(real_bsp, real_hsp)
            pred_fake = D(fake_bsp.detach(), fake_hsp.detach())

            valid = torch.ones_like(pred_real) * 0.9
            fake = torch.zeros_like(pred_fake) + 0.1

            loss_D_real = criterion_GAN(pred_real, valid)
            loss_D_fake = criterion_GAN(pred_fake, fake)
            loss_D = (loss_D_real + loss_D_fake) / 2

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            pred_fake = D(fake_bsp, fake_hsp)
            valid = torch.ones_like(pred_fake) * 0.9
            loss_GAN = criterion_GAN(pred_fake, valid)
            loss_L1 = criterion_L1(fake_bsp, real_bsp) + criterion_L1(fake_hsp, real_hsp)
            loss_G = loss_GAN + 10 * loss_L1

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

        print(f"[Epoch {epoch+1}/{epochs}] Loss_G: {loss_G.item():.4f} | Loss_D: {loss_D.item():.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(G.state_dict(), f"scgan_epoch_{epoch+1}.pth")

    torch.save(G.state_dict(), "scgan_generator_tuned.pth")
    print("✅ Model saved to scgan_generator_tuned.pth")

if __name__ == "__main__":
    train_scgan(data_folder="sequence_dataset", epochs=50, batch_size=2, device="cuda" if torch.cuda.is_available() else "cpu")
