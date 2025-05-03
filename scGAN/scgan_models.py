import torch
import torch.nn as nn

class SeqBSPHSPGenerator(nn.Module):
    def __init__(self, latent_dim=100, seq_len=15, out_channels=1, feature_maps=64):
        super().__init__()
        self.seq_len = seq_len
        self.feature_maps = feature_maps
        self.latent_dim = latent_dim

        # Project latent vector into [B, 512, T, 4, 4]
        self.fc = nn.Linear(latent_dim, feature_maps * 8 * seq_len * 4 * 4)

        # Condition encoder: downsample spatially (H, W), but preserve time (T)
        self.condition_encoder = nn.Sequential(
            nn.Conv3d(2, feature_maps * 2, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),  # 128x128 -> 64x64
            nn.LeakyReLU(0.2),
            nn.Conv3d(feature_maps * 2, feature_maps * 4, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),  # 64x64 -> 32x32
            nn.LeakyReLU(0.2),
            nn.Conv3d(feature_maps * 4, feature_maps * 8, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),  # 32x32 -> 16x16
            nn.LeakyReLU(0.2),
            nn.Conv3d(feature_maps * 8, feature_maps * 8, kernel_size=(1, 4, 4), stride=(1, 4, 4), padding=(0, 0, 0)),  # 16x16 -> 4x4
            nn.LeakyReLU(0.2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(feature_maps * 16, feature_maps * 4,
                               kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(feature_maps * 4),
            nn.ReLU(True),

            nn.ConvTranspose3d(feature_maps * 4, feature_maps * 2,
                               kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(feature_maps * 2),
            nn.ReLU(True),

            nn.ConvTranspose3d(feature_maps * 2, feature_maps,
                               kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(feature_maps),
            nn.ReLU(True),

            nn.ConvTranspose3d(feature_maps, feature_maps // 2,
                               kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(feature_maps // 2),
            nn.ReLU(True),

            nn.ConvTranspose3d(feature_maps // 2, out_channels * 2,
                               kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.Tanh()
        )

    def forward(self, z, condition):
        """
        z: [B, latent_dim]
        condition: [B, 2, T, 128, 128]  # BSP + HSP
        """
        B = z.size(0)

        assert condition.shape[1] == 2, f"Expected 2 channels, got {condition.shape[1]}"
        assert condition.dim() == 5, f"Expected 5D input [B, 2, T, H, W], got {condition.shape}"


        # Project noise
        z_feat = self.fc(z).view(B, self.feature_maps * 8, self.seq_len, 4, 4)  # [B, 512, T, 4, 4]

        # Encode condition
        c_feat = self.condition_encoder(condition)  # [B, 512, T, 4, 4]

        # Merge latent and conditional features
        x = torch.cat([z_feat, c_feat], dim=1)  # [B, 1024, T, 4, 4]

        output = self.decoder(x)  # [B, 2, T, 128, 128]

        bsp, hsp = torch.chunk(output, 2, dim=1)  # split into BSP and HSP

        return bsp, hsp

# -------------------------------
# Discriminator: BSP-HSP Sequence Pair
# -------------------------------
class SeqBSPHSPDiscriminator(nn.Module):
    def __init__(self, in_channels=4, feature_maps=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv3d(in_channels, feature_maps, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(feature_maps, feature_maps * 2, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(feature_maps * 2, feature_maps * 4, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(feature_maps * 4, 1, kernel_size=(1, 4, 4)),
            nn.Sigmoid()
        )

    def forward(self, raw_bsp, raw_hsp):
        def laplacian(x):
            d2x = torch.gradient(torch.gradient(x, dim=3)[0], dim=3)[0]
            d2y = torch.gradient(torch.gradient(x, dim=4)[0], dim=4)[0]
            return d2x+d2y
        
        lap_bsp = laplacian(raw_bsp)
        lap_hsp = laplacian(raw_hsp)

        #Combine into one tensor
        x = torch.cat([raw_bsp, lap_bsp, raw_hsp, lap_hsp], dim=1)
        return self.model(x)
