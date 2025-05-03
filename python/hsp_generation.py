import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import argparse
from matplotlib.image import imread

def compute_hspm(b, A, lam=0.01):
    AT = A.T
    reg_matrix = lam * np.eye(A.shape[1])
    h = np.linalg.inv(AT @ A + reg_matrix) @ AT @ b
    return h

def main():
    parser = argparse.ArgumentParser(description="Generate HSPM from a BSPM using Tikhonov regularization.")
    parser.add_argument("--bspm", type=str, required=True,
                        help="bsp_heatmaps_real_geometry/1/Subject20_run01/bsp_0000.png")
    parser.add_argument("--forward", type=str, required=True,
                        help="matlab/forward_matrix_ep.mat")
    # change '--lambda' to '--lam' or another non-keyword
    parser.add_argument("--lam", type=float, default=0.01,
                        help="Regularization parameter (default: 0.01).")
    parser.add_argument("--output", type=str, default="generated_HSPM.png",
                        help="hspm")
    args = parser.parse_args()

    # Load forward matrix
    mat_data = sio.loadmat(args.forward)
    if 'forward_matrix_ep' in mat_data:
        A = mat_data['forward_matrix_ep']
    else:
        raise ValueError("Forward matrix 'A' not found in the file.")

    # Load BSPM
    try:
        bspm_array = np.load(args.bspm)
        b = bspm_array.flatten()
    except Exception:
        bspm_img = imread(args.bspm)
        if bspm_img.ndim == 3:
            bspm_img = np.mean(bspm_img, axis=2)
        b = bspm_img.flatten()

    # Compute HSPM using Tikhonov regularization
    h_vector = compute_hspm(b, A, args.lam)

    # Reshape the result (this is just an example—actual reshaping depends on your heart mesh)
    n_nodes = h_vector.shape[0]
    side = int(np.sqrt(n_nodes))
    if side * side != n_nodes:
        hsp_image = h_vector.reshape((-1, 1))
    else:
        hsp_image = h_vector.reshape((side, side))

    # Display and save
    plt.imshow(hsp_image, cmap='jet')
    plt.title('Generated HSPM')
    plt.colorbar()
    plt.savefig(args.output)
    print(f"HSPM has been generated and saved to {args.output}")

if __name__ == '__main__':
    main()