import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scgan_models import SeqBSPHSPGenerator

class PairedSequenceDataset(Dataset):
    def __init__(self, folder, seq_len=15):
        self.files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.npz')])
        self.seq_len = seq_len

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        bsp = torch.tensor(data["bsp"], dtype=torch.float32)
        hsp = torch.tensor(data["hsp"], dtype=torch.float32)
        return bsp, hsp

def pretrain_generator(data_folder, epochs=10, batch_size=2, latent_dim=100, device="cuda"):
    dataset = PairedSequenceDataset(data_folder)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    G = SeqBSPHSPGenerator(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = nn.L1Loss()

    for epoch in range(epochs):
        for i, (real_bsp, real_hsp) in enumerate(dataloader):
            real_bsp = real_bsp.to(device).permute(0, 2, 1, 3, 4)
            real_hsp = real_hsp.to(device).permute(0, 2, 1, 3, 4)

            z = torch.randn(real_bsp.size(0), latent_dim, device=device)
            fake_bsp, fake_hsp = G(z)

            loss = criterion(fake_bsp, real_bsp) + criterion(fake_hsp, real_hsp)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[Epoch {epoch+1}/{epochs}] L1 Pretrain Loss: {loss.item():.4f}")

    torch.save(G.state_dict(), "generator_l1_pretrained.pth")
    print("✅ Pretrained generator saved to generator_l1_pretrained.pth")

if __name__ == "__main__":
    pretrain_generator(data_folder="sequence_dataset", epochs=10, batch_size=2, device="cuda" if torch.cuda.is_available() else "cpu")
