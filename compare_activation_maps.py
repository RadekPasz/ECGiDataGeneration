import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from skimage.transform import resize
from glob import glob

# === CONFIG ===
REAL_DIR = "sequence_dataset"
SYNTH_DIR = "generated_npz"
OUT_CSV = "activation_comparison_results.csv"
SHAPE_TARGET = (64, 64)

def compute_activation_times(hsp_seq):
    T, H, W = hsp_seq.shape
    at_map = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            v = hsp_seq[:, i, j]
            dv = np.gradient(v)
            at_map[i, j] = np.argmin(dv)
    return at_map

def load_hsp(path, key='hsp', target_shape=SHAPE_TARGET):
    data = np.load(path)
    hsp = np.squeeze(data[key])  # (T, H, W)
    if hsp.shape[1:] != target_shape:
        hsp = np.array([resize(frame, target_shape, mode='reflect', anti_aliasing=True) for frame in hsp])
    return hsp

# === MAIN ===
real_files = sorted(glob(os.path.join(REAL_DIR, "pair_*.npz")))
synth_files = sorted(glob(os.path.join(SYNTH_DIR, "sample_*.npz")))

results = []
for rfile, sfile in zip(real_files, synth_files):
    try:
        hsp_real = load_hsp(rfile)
        hsp_synth = load_hsp(sfile)

        at_real = compute_activation_times(hsp_real)
        at_synth = compute_activation_times(hsp_synth)

        cc = pearsonr(at_real.flatten(), at_synth.flatten())[0]
        mae = np.mean(np.abs(at_real - at_synth))

        results.append({
            "real_file": os.path.basename(rfile),
            "synthetic_file": os.path.basename(sfile),
            "pearson_cc": cc,
            "mae": mae
        })
    except Exception as e:
        results.append({
            "real_file": os.path.basename(rfile),
            "synthetic_file": os.path.basename(sfile),
            "error": str(e)
        })

# Save results
df = pd.DataFrame(results)
df.to_csv(OUT_CSV, index=False)
print(f"Done. Results saved to: {OUT_CSV}")
