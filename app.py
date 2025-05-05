import os
import torch
from scGAN.scgan_train import train_scgan  # adjust path if needed

MODEL_PATH = "scgan_epoch_40.pth"  # adjust to your target model file
DATA_PATH = "sequence_dataset"     # must match what's present in the image

def model_already_trained(path):
    return os.path.exists(path)

def remove_temp_files():
    # Customize this if you're saving temp images or logs
    temp_dirs = ['generated_gifs', 'generated_samples', 'generated_npz']
    for dir in temp_dirs:
        if os.path.exists(dir):
            print(f"Cleaning up: {dir}")
            for file in os.listdir(dir):
                os.remove(os.path.join(dir, file))

if __name__ == "__main__":
    print("=== ECGi Trainer ===")
    
    if model_already_trained(MODEL_PATH):
        print(f"Model {MODEL_PATH} already exists. Skipping training.")
    else:
        print("Starting training...")
        try:
            train_scgan(DATA_PATH)
            print("Training complete.")
        except Exception as e:
            print(f"Training failed: {e}")
    
    remove_temp_files()
    print("Cleanup done. Exiting.")
