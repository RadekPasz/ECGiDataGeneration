import os
import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm

# Settings
input_folder = "sequence_dataset"
output_folder = "resized_sequence_dataset"
target_len = 150
target_size = (128, 128)

os.makedirs(output_folder, exist_ok=True)

def resize_sequence(tensor, seq_len=150, size=(128, 128)):
    """
    Expects shape: (T, 1, H, W) or (1, T, H, W)
    Returns: resized tensor with shape (T, 1, H, W)
    """
    tensor = torch.tensor(tensor, dtype=torch.float32)
    if tensor.dim() == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)  # (T, H, W)
    elif tensor.dim() == 4 and tensor.shape[1] == 1:
        tensor = tensor.squeeze(1)  # (T, H, W)
    tensor = tensor.unsqueeze(1)  # (T, 1, H, W)
    tensor = tensor.permute(1, 0, 2, 3).unsqueeze(0)  # (1, 1, T, H, W)
    resized = F.interpolate(tensor, size=(seq_len, *size), mode="trilinear", align_corners=False)
    return resized.squeeze(0).permute(1, 0, 2, 3).numpy()  # (T, 1, H, W)

for fname in tqdm(os.listdir(input_folder)):
    if not fname.endswith(".npz"):
        continue

    data = np.load(os.path.join(input_folder, fname))
    bsp_resized = resize_sequence(data["bsp"], seq_len=target_len, size=target_size)
    hsp_resized = resize_sequence(data["hsp"], seq_len=target_len, size=target_size)

    out_path = os.path.join(output_folder, fname)
    np.savez_compressed(out_path, bsp=bsp_resized, hsp=hsp_resized)

print("All sequences resized and saved to", output_folder)
