import os
import numpy as np
import matplotlib.pyplot as plt
import random

real_dir = "sequence_dataset"
fake_dir = "generated_npz"

#Pick one real and one fake sequence
real_path = sorted([f for f in os.listdir(real_dir) if f.endswith(".npz")])[0]
fake_npz_path = sorted(os.listdir("generated_npz"))[0]

real = np.load(os.path.join(real_dir, real_path), allow_pickle=True)
fake = np.load(os.path.join("generated_npz", fake_npz_path), allow_pickle=True)

print("Type of real['bsp']:", type(real["bsp"]))
print("real['bsp']:", real["bsp"])
print("Shape of real['bsp']:", real["bsp"].shape)
bsp_real = real["bsp"][:, 0] 

print("Shape of fake['bsp']:", fake["bsp"].shape)
bsp_fake = fake["bsp"][0, 0]  

pixels_to_check = [(32, 32), (10, 50), (45, 20), (0, 0), (10, 10)]
#std_map_real = bsp_real.std(axis=0)  
#max_idx = np.unravel_index(np.argmax(std_map_real), std_map_real.shape)

#pixels_to_check = [max_idx]
#print(f"Pixel with most variation: {max_idx}")

for y, x in pixels_to_check:
    plt.figure(figsize=(5, 2))
    plt.plot(bsp_real[:, y, x], label="Real", linewidth=2)
    plt.plot(bsp_fake[:, y, x], label="Fake", linestyle='--')
    plt.title(f"Temporal Signal at pixel (x={x}, y={y})")
    plt.xlabel("Time step")
    plt.ylabel("Potential")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
