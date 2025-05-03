import os
import numpy as np
from glob import glob
from PIL import Image

# === CONFIG ===
bsp_root = "bsp_heatmaps_real_geometry"
hsp_root = "hspm"
output_dir = "sequence_dataset"
os.makedirs(output_dir, exist_ok=True)

# Maps BSP folders to HSP pacing folders
hsp_pace_map = {
    "1": "InterventionPace1",
    "2": "InterventionPace2",
    "3": "InterventionPace3",
    "4": "InterventionPace4",
    "5": "InterventionPace5",
    "6": "InterventionPace6",
    "7": "InterventionPace7",
}

# === IMAGE LOADING FUNCTION ===
def load_image_sequence(folder, pattern, length=150):
    full_pattern = os.path.join(folder, pattern).replace("\\", "/")
    image_paths = sorted(glob(full_pattern))[:length]
    frames = []
    for path in image_paths:
        img = Image.open(path).convert("L")
        frames.append(np.array(img, dtype=np.float32) / 255.0)
    return np.stack(frames, axis=0)[:, None, :, :]  # (T, 1, H, W)

# === MAIN LOOP ===
index = 0
for bsp_key in os.listdir(bsp_root):
    bsp_path = os.path.join(bsp_root, bsp_key)
    hsp_folder = os.path.join(hsp_root, hsp_pace_map.get(bsp_key, ""))
    if not os.path.isdir(bsp_path) or not os.path.isdir(hsp_folder):
        continue

    for run in os.listdir(bsp_path):
        bsp_seq_path = os.path.join(bsp_path, run)
        hsp_seq_path = hsp_folder  # HSP files are flat, not in subfolders

        try:
            # Load BSP (from its own folder)
            bsp_seq = load_image_sequence(bsp_seq_path, pattern="bsp_*.png")
            
            # Load HSP (directly from the InterventionPace folder)
            hsp_pattern = f"{run}_t????_hsp_heatmap.png"
            hsp_seq = load_image_sequence(hsp_seq_path, pattern=hsp_pattern)

            if bsp_seq is None or hsp_seq is None:
                continue

            out_path = os.path.join(output_dir, f"pair_{index:04d}.npz")
            np.savez_compressed(out_path, bsp=bsp_seq, hsp=hsp_seq)
            print(f"✅ Saved: {out_path}")
            index += 1

        except Exception as e:
            print(f"❌ Error in {run}: {e}")

print(f"\n🎉 Done. Saved {index} valid samples to '{output_dir}/'")
