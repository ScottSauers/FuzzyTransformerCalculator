import os
import pickle
import numpy as np

input_file_path = "/Users/scott/Downloads/Jax/math_problems_dataset.txt"
output_dir = "/Users/scott/Downloads/Jax"

pad_token = 17
def encode_and_pad_problem_solution(line, stoi, problem_length=8, solution_length=8, pad_token=17, equals_token=16):
    """Encodes a line into problem and solution parts, padding each part accordingly."""
    problem_part, solution_part = line.split('=')
    # Encode each part
    encoded_problem = [stoi.get(c, pad_token) for c in problem_part.strip()]
    encoded_solution = [stoi.get(c, pad_token) for c in solution_part.strip()]
    # Pad each part independently
    padded_problem = np.pad(encoded_problem, (0, problem_length - len(encoded_problem)),
                            mode='constant', constant_values=pad_token)
    padded_solution = np.pad(encoded_solution, (0, solution_length - len(encoded_solution)),
                             mode='constant', constant_values=pad_token)
    # Concatenate padded parts
    return np.concatenate([padded_problem, [equals_token], padded_solution], axis=0)

def preprocess_and_save(input_file_path, output_dir, stoi, problem_length=8, solution_length=8):
    all_data = []
    with open(input_file_path, 'r') as f:
        for line in f:
            all_data.append(encode_and_pad_problem_solution(line, stoi, problem_length, solution_length))

    # Shuffle the data to ensure a random distribution between train and validation sets
    np.random.shuffle(all_data)
    all_data = np.array(all_data)

    # Split into training and validation sets
    split_idx = int(len(all_data) * 0.9)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]

    # Save the datasets as .npy files for efficient loading during training
    np.save(os.path.join(output_dir, 'train.npy'), train_data)
    np.save(os.path.join(output_dir, 'val.npy'), val_data)

# Hardcoded stoi and itos dictionaries
stoi = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '\\n': 10, '*': 11, '+': 12, '-': 13, '.': 14, '/': 15, '=': 16, '<pad>': 17}
itos = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '\\n', 11: '*', 12: '+', 13: '-', 14: '.', 15: '/', 16: '=', 17: '<pad>'}
vocab_size = len(stoi)

# Run the preprocessing and save the datasets
preprocess_and_save(input_file_path, output_dir, stoi, problem_length=8, solution_length=8)

# Save the meta information as well
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("All unique characters in model:", ''.join(itos.values()))
print("Character to index mapping:", stoi)

# Load a small sample from the saved training data to print example tokens and decoded text
train_data_sample = np.load(os.path.join(output_dir, 'train.npy'))[:5]  # Load first 5 examples
for i, sample in enumerate(train_data_sample):
    problem_solution_text = ''.join(meta['itos'][int(idx)] for idx in sample if idx != pad_token)  # Decode entire sequence
    print(f"Example {i+1}: Encoded Tokens: {sample}, Decoded Text: '{problem_solution_text}'")