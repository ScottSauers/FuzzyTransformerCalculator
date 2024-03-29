import os
import numpy as np
import pickle

# Set the directory where your files are saved
output_dir = "/Users/scott/Downloads/Jax"

# Load the metadata
meta_path = os.path.join(output_dir, 'meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
itos = meta['itos']

# Load the preprocessed training data
train_data_path = os.path.join(output_dir, 'train.npy')
train_data = np.load(train_data_path)

# Decode and print the first 25 lines
for i, sample in enumerate(train_data[:25]):
    decoded_text = ''.join(itos[idx] for idx in sample if idx != 42)  # Decode each token, ignoring padding
    print(f"Line {i+1}: {decoded_text}")