
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

def load_forward_matrix(path):
    data = scipy.io.loadmat(path)
    return data['forward_matrix_ep']  

def load_bsp_frame(mat_path, frame_idx=0):
    data = scipy.io.loadmat(mat_path)
    bsp_struct = data['bspm'][0, 0]
    bsp = bsp_struct['potvals']
    return bsp[:, frame_idx]  

def tikhonov_inverse(A, b, lam=0.01):
    AtA = A.T @ A
    reg = lam * np.eye(AtA.shape[0])
    A_pinv = np.linalg.inv(AtA + reg) @ A.T
    return A_pinv @ b  # HSP estimate

def normalize_and_plot(hsp, save_path=None):
    scaler = MinMaxScaler()
    hsp_reshaped = hsp.reshape(-1, 1)
    hsp_scaled = scaler.fit_transform(hsp_reshaped).flatten()

    plt.figure(figsize=(12, 2))
    plt.plot(hsp_scaled)
    plt.title("Estimated Epicardial Potentials (HSP)")
    plt.xlabel("Epicardial node index")
    plt.ylabel("Normalized potential")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

if __name__ == "__main__":
    forward_matrix_path = "python/forward_matrix_ep.mat"
    bsp_mat_path = "C:/Users/radek/OneDrive/Pulpit/Thesis/Interventions/InterventionPace1/Subject20_run01.mat"
    output_plot_path = "estimated_hsp.png"
    timestep = 83  # midpoint

    print("Loading forward matrix...")
    A = load_forward_matrix(forward_matrix_path)

    print(f"Loading BSP frame {timestep}...")
    bsp_frame = load_bsp_frame(bsp_mat_path, frame_idx=timestep)

    print("Computing inverse using Tikhonov regularization...")
    hsp_est = tikhonov_inverse(A, bsp_frame, lam=0.01)

    scipy.io.savemat("estimated_hsp.mat", {"hsp_est": hsp_est})
    print("Saved estimated HSP to estimated_hsp.mat")

    print(f"Plotting and saving result to {output_plot_path}...")
    normalize_and_plot(hsp_est, save_path=output_plot_path)
