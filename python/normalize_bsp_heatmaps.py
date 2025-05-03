import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def normalize_and_save(folder_path, output_folder=None):
    # Gather all PNG files
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
    if not image_files:
        print(f"No PNGs found in {folder_path}")
        return

    images = []
    for fname in image_files:
        img = mpimg.imread(os.path.join(folder_path, fname))
        images.append(img)

    all_pixels = np.concatenate([img.flatten() for img in images])
    global_min = np.min(all_pixels)
    global_max = np.max(all_pixels)

    print(f"Normalizing {len(images)} images: min={global_min:.3f}, max={global_max:.3f}")

    if output_folder is None:
        output_folder = folder_path  # overwrite in-place
    else:
        os.makedirs(output_folder, exist_ok=True)

    for i, fname in enumerate(image_files):
        img = images[i]
        norm_img = (img - global_min) / (global_max - global_min + 1e-9)
        norm_img = np.clip(norm_img, 0, 1)
        save_path = os.path.join(output_folder, fname)
        plt.imsave(save_path, norm_img, cmap='jet')

    print(f"Saved normalized images to {output_folder}")

if __name__ == "__main__":
    input_folder = "bsp_heatmaps_struct/1/Subject20_run01"
    output_folder = "normalized_bsp"

    normalize_and_save(input_folder, output_folder)
