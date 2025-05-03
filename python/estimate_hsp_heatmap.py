import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from sklearn.decomposition import PCA
import os
from scipy.interpolate import RBFInterpolator

def load_forward_matrix(path):
    data = scipy.io.loadmat(path)
    return data['forward_matrix_ep']  

def load_bsp_frame(mat_path, frame_idx=0):
    data = scipy.io.loadmat(mat_path)
    bsp_struct = data['bspm'][0, 0]
    bsp = bsp_struct['potvals']
    return bsp[:, frame_idx]  

def load_epicardial_geometry(geom_path):
    data = scipy.io.loadmat(geom_path)
    return data['heart_geom']['node'][0, 0]  

def tikhonov_inverse(A, b, lam=0.01):
    AtA = A.T @ A
    reg = lam * np.eye(AtA.shape[0])
    A_pinv = np.linalg.inv(AtA + reg) @ A.T
    return A_pinv @ b  

def forward_projection(A, h):
    return A @ h

def compute_forward_error(b_orig, b_est):
    corr = np.corrcoef(b_orig, b_est)[0, 1]
    mae = np.mean(np.abs(b_orig - b_est))
    return corr, mae

def save_hsp_heatmap(hsp, coords, save_path, grid_size=256, use_pca=True):
    if use_pca:
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(coords)  
        x = coords_2d[:, 0]
        y = coords_2d[:, 1]
    else:
        x = coords[:, 0]
        y = coords[:, 1]
        x -= np.mean(x)
        y -= np.mean(y)
    
    # Normalize the 2D coordinates to roughly be in [-1, 1]
    scale = max(np.max(np.abs(x)), np.max(np.abs(y)))
    if scale != 0:
        x /= scale
        y /= scale
    
    # Normalize HSP values to [0, 1]
    hsp_norm = (hsp - np.min(hsp)) / (np.max(hsp) - np.min(hsp) + 1e-9)
    
    # Create a uniform interpolation grid
    xi = yi = np.linspace(-1, 1, grid_size)
    xx, yy = np.meshgrid(xi, yi)
    
    # Interpolate HSP values onto the regular grid
    rbf = RBFInterpolator(np.column_stack((x, y)), hsp_norm, smoothing=0.1)
    zz = rbf(np.column_stack((xx.ravel(), yy.ravel()))).reshape(xx.shape)
    
    # Create a mask of the convex hull of the points using nearest neighbor interpolation
    mask = griddata((x, y), np.ones_like(hsp_norm), (xx, yy), method='nearest', fill_value=0)
    
    vmin = np.nanpercentile(zz, 5)
    vmax = np.nanpercentile(zz, 95)
    
    plt.figure(figsize=(5, 5))
    plt.imshow(zz, cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__ == "__main__":
    forward_matrix_path = "matlab/forward_matrix_ep.mat"
    bsp_input_folder = "Interventions" 
    geometry_path = "matlab/heart_geom.mat"
    output_base_folder = "hspm"         # Output folder 

    print("Loading forward matrix...")
    A = load_forward_matrix(forward_matrix_path)

    print("Loading epicardial geometry...")
    epicardial_nodes = load_epicardial_geometry(geometry_path)

    # Walk through all BSP files in the input folder recursively.
    for pace in range(2, 8):  # InterventionPace2 to 7
        subfolder = f"InterventionPace{pace}"
        pace_path = os.path.join(bsp_input_folder, subfolder)
        if not os.path.isdir(pace_path):
            continue

        for file in os.listdir(pace_path):
            if not file.endswith("_run01.mat"):
                continue

            bsp_mat_path = os.path.join(pace_path, file)
            bsp_mat_path = os.path.join(pace_path, file)
            print(f"Processing BSP file: {bsp_mat_path}")
            try:
                data = scipy.io.loadmat(bsp_mat_path)
                bsp_struct = data['bspm'][0, 0]
                bsp = bsp_struct['potvals']  
            except Exception as e:
                print(f"Error loading BSP from {bsp_mat_path}: {e}")
                continue

            n_timesteps = bsp.shape[1]
            print(f"Found {n_timesteps} timesteps.")

            for t in range(n_timesteps):
                print(f"Processing timestep {t}...")
                bsp_frame = bsp[:, t]

                print("Computing inverse using Tikhonov regularization...")
                hsp_est = tikhonov_inverse(A, bsp_frame, lam=0.01)

                print("Performing forward projection check...")
                bsp_est = forward_projection(A, hsp_est)
                corr, mae = compute_forward_error(bsp_frame, bsp_est)
                print(f"  Forward check: Pearson correlation = {corr:.4f}, MAE = {mae:.4f}")

                # Define output folder path
                rel_path = os.path.relpath(pace_path, bsp_input_folder)
                output_folder = os.path.join(output_base_folder, rel_path)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                base_filename = os.path.splitext(file)[0]
                output_image_path = os.path.join(output_folder, f"{base_filename}_t{t:04d}_hsp_heatmap.png")

                print(f"Saving HSP heatmap to {output_image_path}...")
                save_hsp_heatmap(hsp_est, epicardial_nodes, output_image_path, grid_size=256, use_pca=True)
                print(f"  Saved HSP heatmap to {output_image_path}\n")

    print("Processing complete!")

