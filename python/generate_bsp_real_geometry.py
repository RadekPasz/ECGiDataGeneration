import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from sklearn.preprocessing import MinMaxScaler

def tps_interpolate(xyz, values, grid_size=128):
    x, y = xyz[:, 0], xyz[:, 1]
    z = values
    rbf = Rbf(x, y, z, function='thin_plate')
    xi = yi = np.linspace(np.min(x), np.max(x), grid_size)
    xx, yy = np.meshgrid(xi, yi)
    zz = rbf(xx, yy)
    return zz

def load_electrode_positions(path):
    data = scipy.io.loadmat(path)
    return data['electrode_positions']['node'][0, 0]  

def process_struct_bsp(mat_path, save_dir, torso_xyz, grid_size=128):
    data = scipy.io.loadmat(mat_path)

    if 'bspm' not in data:
        print(f"No 'bspm' in {mat_path}")
        return

    bsp_struct = data['bspm'][0, 0]
    
    if 'potvals' not in bsp_struct.dtype.names:
        print(f"No 'potvals' field in bspm struct of {mat_path}")
        return

    bsp = bsp_struct['potvals']  

    if bsp.size == 0 or bsp.shape[0] < 3:
        print(f"BSP data is empty or insufficient in {mat_path}")
        return

    os.makedirs(save_dir, exist_ok=True)

    for t in range(bsp.shape[1]):
        bsp_vals = bsp[:, t]

        if np.isnan(bsp_vals).any():
            print(f"Skipping t={t} due to NaNs")
            continue

        try:
            bsp_img = tps_interpolate(torso_xyz, bsp_vals, grid_size)
            scaler = MinMaxScaler()
            bsp_img = scaler.fit_transform(bsp_img)
            plt.imsave(os.path.join(save_dir, f"bsp_{t:04d}.png"), bsp_img, cmap='jet')
        except Exception as e:
            print(f"Error at t={t}: {e}")

if __name__ == "__main__":
    base_root = "Interventions"
    save_root = "bsp_heatmaps_real_geometry"
    electrode_path = "matlab/electrode_positions.mat"

    torso_xyz = load_electrode_positions(electrode_path)
    os.makedirs(save_root, exist_ok=True)

    for pace_folder in sorted(os.listdir(base_root)):
        pace_path = os.path.join(base_root, pace_folder)

        if not os.path.isdir(pace_path) or not pace_folder.startswith("InterventionPace"):
            continue

        pace_index = pace_folder.replace("InterventionPace", "")
        intervention_output_dir = os.path.join(save_root, pace_index)
        os.makedirs(intervention_output_dir, exist_ok=True)

        for file in os.listdir(pace_path):
            if file.endswith(".mat"):
                full_path = os.path.join(pace_path, file)
                sim_name = os.path.splitext(file)[0]
                save_path = os.path.join(intervention_output_dir, sim_name)
                os.makedirs(save_path, exist_ok=True)
                print(f"Generating BSP heatmaps for {sim_name} in InterventionPace{pace_index}...")
                process_struct_bsp(full_path, save_path, torso_xyz)
