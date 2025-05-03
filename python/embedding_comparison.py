import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn.decomposition import PCA

def load_simple_embedding(path, key="hsp", fixed_frames=150):
    data = np.load(path)[key]
    if data.ndim == 5:
        data = data[:, 0, :, :, :]  # (T, H, W)
    elif data.ndim == 4:
        data = np.squeeze(data)
    elif data.ndim == 3:
        pass
    else:
        raise ValueError(f"Unexpected shape: {data.shape}")

    T = data.shape[0]
    mean_per_frame = data.reshape(T, -1).mean(axis=1)  # shape (T,)

    # Pad or truncate to fixed length
    if T < fixed_frames:
        pad = np.zeros(fixed_frames - T)
        mean_per_frame = np.concatenate([mean_per_frame, pad])
    elif T > fixed_frames:
        mean_per_frame = mean_per_frame[:fixed_frames]

    return mean_per_frame

def collect_embeddings(folder, label, max_samples=100):
    files = sorted(glob(os.path.join(folder, "*.npz")))[:max_samples]
    embeddings, labels = [], []
    for f in files:
        try:
            emb = load_simple_embedding(f)
            embeddings.append(emb)
            labels.append(label)
        except Exception as e:
            print(f"❌ Failed on {f}: {e}")
    return embeddings, labels

# === MAIN ===
real_dir = "sequence_dataset"
fake_dir = "generated_npz"

real_emb, real_labels = collect_embeddings(real_dir, "Real", 100)
fake_emb, fake_labels = collect_embeddings(fake_dir, "Fake", 100)

all_emb = np.vstack(real_emb + fake_emb)
all_labels = real_labels + fake_labels

# Reduce with PCA (faster & safer than t-SNE)
pca = PCA(n_components=2)
emb_2d = pca.fit_transform(all_emb)

# Plot
plt.figure(figsize=(8, 6))
for label in set(all_labels):
    idx = [i for i, l in enumerate(all_labels) if l == label]
    plt.scatter(emb_2d[idx, 0], emb_2d[idx, 1], label=label, alpha=0.7)
plt.title("PCA of Real vs Synthetic (Mean-Per-Frame Embeddings)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
