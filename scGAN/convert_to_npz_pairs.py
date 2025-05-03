import os
import numpy as np
from PIL import Image
from torchvision import transforms
import torch

# Configuration
bsp_dir = "bsp_heatmaps_real_geometry/1/Subject20_run01"
hsp_dir = "hspm/InterventionPace1/Subject20_run01"
output_dir = "sequence_dataset"
sequence_length = 15

os.makedirs(output_dir, exist_ok=True)

# Image transformation: Resize and normalize to [-1, 1]
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Sorted list of frame filenames
bsp_frames = sorted([f for f in os.listdir(bsp_dir) if f.endswith('.png')])
hsp_frames = sorted([f for f in os.listdir(hsp_dir) if f.endswith('.png')])

assert len(bsp_frames) == len(hsp_frames), "Mismatch in BSP and HSP frame counts."

# Sliding window over sequence
pair_id = 0
for i in range(len(bsp_frames) - sequence_length + 1):
    bsp_seq = []
    hsp_seq = []
    for j in range(sequence_length):
        bsp_img = Image.open(os.path.join(bsp_dir, bsp_frames[i + j])).convert("L")
        hsp_img = Image.open(os.path.join(hsp_dir, hsp_frames[i + j])).convert("L")
        bsp_tensor = transform(bsp_img).numpy()
        hsp_tensor = transform(hsp_img).numpy()
        bsp_seq.append(bsp_tensor)
        hsp_seq.append(hsp_tensor)

    bsp_array = np.stack(bsp_seq, axis=0)  # [T, 1, 128, 128]
    hsp_array = np.stack(hsp_seq, axis=0)

    out_path = os.path.join(output_dir, f"pair_{pair_id:04d}.npz")
    np.savez(out_path, bsp=bsp_array, hsp=hsp_array)
    pair_id += 1

print(f"Created {pair_id} BSP-HSP pairs in {output_dir}")
