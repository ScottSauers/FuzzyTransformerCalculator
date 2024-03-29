import numpy as np
import os
import pickle
from flax import linen as nn
from flax.training import train_state
import optax
import jax
from jax import numpy as random, jit, grad
from jax import value_and_grad
import jax.numpy as jnp
from jax.nn import log_softmax
from jax import vmap
from typing import Sequence
from torch.utils.data import DataLoader, Dataset
from jax import random
from jax import tree_util
import glob
from to_csv import LossLogger
import datetime
from datetime import datetime
from jax.nn import one_hot
from jax.scipy.special import logsumexp
from jax.nn import softmax
from jax.experimental.host_callback import id_print
import jax.lax as lax

from loss import compute_loss

logger = LossLogger()



# Assuming meta.pkl contains itos mapping
with open('meta.pkl', 'rb') as f:
    meta = pickle.load(f)
itos = meta['itos']
stoi = meta['stoi']
vocab_size = meta['vocab_size']


# Data Loading
def load_data_npy(file_path):
    """Loads data as a memory-mapped array."""
    return np.load(file_path, mmap_mode='r')


current_dir = os.getcwd()
train_data_path = os.path.join(current_dir, 'train.npy')
val_data_path = os.path.join(current_dir, 'val.npy')
train_data = load_data_npy(train_data_path)
val_data = load_data_npy(val_data_path)


# Transformer Model
class PositionalEncoding(nn.Module):
    d_model: int
    max_len: int = 5000

    def setup(self):
        key = random.PRNGKey(0)

        # Use jax.random.normal to initialize self.pe with small random values
        self.pe = random.normal(key, (self.max_len, self.d_model)) * 0.01

        position = jnp.arange(0, self.max_len).reshape(-1, 1)
        div_term = jnp.exp(jnp.arange(0, self.d_model, 2) * -(jnp.log(10000.0) / self.d_model))

        # Update self.pe with sine and cosine values while keeping the initial random values small and controlled
        self.pe = self.pe.at[:, 0::2].set(jnp.sin(position * div_term))
        self.pe = self.pe.at[:, 1::2].set(jnp.cos(position * div_term))

    def __call__(self, x):
        pe = self.pe[None, :x.shape[1], :]  # Add a batch dimension
        return x + pe


class TransformerBlock(nn.Module):
    d_model: int
    num_heads: int
    d_ff: int
    dropout_rate: float = 0

    @nn.compact
    def __call__(self, x, training):
        # Pass deterministic flag correctly based on the training status
        attn_out = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            dropout_rate=self.dropout_rate,
            deterministic=not training  # Correctly control stochastic behavior based on training status
        )(x)
        attn_out = nn.LayerNorm()(attn_out + x)

        ff_out = nn.Dense(self.d_ff)(attn_out)
        ff_out = nn.relu(ff_out)
        ff_out = nn.Dense(self.d_model)(ff_out)
        # Also, make sure dropout knows whether it's training or not
        ff_out = nn.Dropout(
            rate=self.dropout_rate,
            deterministic=not training  # Ensure dropout is only applied during training
        )(ff_out)

        return nn.LayerNorm()(ff_out + attn_out)



class Transformer(nn.Module):
    vocab_size: int
    d_model: int = 32
    num_heads: int = 4
    num_layers: int = 6
    d_ff: int = 128

    def setup(self):
        self.token_embedding = nn.Embed(self.vocab_size, self.d_model)
        self.pos_encoding = PositionalEncoding(d_model=self.d_model)
        self.transformer_blocks = [TransformerBlock(self.d_model, self.num_heads, self.d_ff) for _ in range(self.num_layers)]
        self.dense_output = nn.Dense(1)  # Predict a single value representing the solution


    def __call__(self, input_ids, training=True):
        x = self.token_embedding(input_ids)
        #id_print(x, what="Embed")
        #id_print(jnp.any(jnp.isnan(x)), what="Embed NaNs")

        x = self.pos_encoding(x)
        #id_print(x, what="PosEnc")
        #id_print(jnp.any(jnp.isnan(x)), what="PosEnc NaNs")

        for i, block in enumerate(self.transformer_blocks):
            x = block(x, training)
            #id_print(x, what=f"Block {i+1}")
            #id_print(jnp.any(jnp.isnan(x)), what=f"Block {i+1} NaNs")

        x = self.dense_output(x)
        id_print(x.squeeze(-1), what="Output")
        #id_print(jnp.any(jnp.isnan(x)), what="Output NaNs")

        return x.squeeze(-1)  # Remove the last dimension to get a 1D array of predicted values


class MathDataset:
    def __init__(self, data_path):
        # Memory-map the .npy file where data is stored
        self.data = np.load(data_path, mmap_mode='r')
        # Define the lengths based on the preprocessing information
        self.problem_length = 8  # Length of the problem part
        self.solution_length = 8  # Length of the solution part
        self.equals_token_index = 16  # '=' is encoded as 16
        self.pad_token = 17
        self.total_length = self.problem_length + 1 + self.solution_length  # Total length including '=' token

    def __len__(self):
        # Return the total number of problems in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Fetch a full sequence (problem + '=' + solution) from the data
        full_sequence = self.data[idx]

        # The problem is everything before the equals token (including it)
        problem = full_sequence[:self.problem_length + 1]  # Including '=' token at the end of the problem

        # The solution is everything after the equals token, excluding padding
        solution = full_sequence[self.problem_length + 1:self.total_length]  # Extract solution part

        # Convert to int32 for compatibility with JAX
        problem = problem.astype(np.int32)
        solution = solution.astype(np.int32)

        # Identify and remove padding from the solution for accurate target representation
        solution = solution[solution != self.pad_token]

        return problem, solution

# Initialize dataset
batch_size = 1

train_dataset = MathDataset(train_data_path)
val_dataset = MathDataset(val_data_path)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Initialize the model and optimizer
rng = random.PRNGKey(0)
input_shape = (32, 25)
model = Transformer(vocab_size=vocab_size)
params = model.init(rng, jnp.ones(input_shape, jnp.int32))['params']
#print("init params: ", params)
nan_params = tree_util.tree_map(lambda x: jnp.any(jnp.isnan(x)), params)
#print("nan params: ", nan_params)
tx = optax.adam(1e-6)

model_dir = 'saved_models'


# Check for the most recent checkpoint directory
if os.path.exists(model_dir):
    checkpoint_dirs = [os.path.join(model_dir, d) for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    latest_checkpoint_dir = max(checkpoint_dirs, key=os.path.getmtime) if checkpoint_dirs else None

    if latest_checkpoint_dir:
        print(f"Loading model from {latest_checkpoint_dir}")
        initial_params = {}
        for param_file in sorted(glob.glob(os.path.join(latest_checkpoint_dir, '*.npy'))):
            param_name = os.path.basename(param_file).replace('param_', '').replace('.npy', '')
            param_value = np.load(param_file)
            initial_params[param_name] = param_value
            print(f"Loaded parameter: {param_name}")

        # Reconstruct the model parameters
        rng = random.PRNGKey(0)
        dummy_input = jnp.ones((1, 25), dtype=jnp.int32)
        initial_structure = model.init(rng, dummy_input)
        _, treedef = tree_util.tree_flatten(initial_structure['params'])
        reconstructed_params = treedef.unflatten([initial_params[str(i)] for i in range(len(initial_params))])
        state = train_state.TrainState.create(apply_fn=model.apply, params=reconstructed_params, tx=tx)
        print("Model loading complete.")
    else:
        print("Checkpoint directory exists but is empty. Starting training from scratch.")
        state = None
else:
    print("No saved model found. Starting training from scratch.")
    os.makedirs(model_dir, exist_ok=True)
    state = None

if state is None:  # If no model was loaded, initialize a new one
    rng = random.PRNGKey(0)
    dummy_input = jnp.ones((1, 25), dtype=jnp.int32)
    params = model.init(rng, dummy_input)['params']
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# Utility functions
#@jit
def train_step(state, inputs, targets, rng_key):
    id_print(inputs, what="Inputs")
    id_print(targets, what="targets")

    def loss_fn(params):

        #id_print(params, what="params")

        # Use the subkey for dropout within this call
        subkey, _ = random.split(rng_key)  # Obtain a subkey
        logits = state.apply_fn({'params': params}, inputs, rngs={'dropout': subkey})

        #id_print(logits, what="logits")

        #print("Getting loss...")
        #print(f"LOGITS: {logits}")
        #print(f"TARGETS: {targets}")
        #print("Logits shape:", logits.shape)
        #print("Targets shape:", targets.shape)
        #print("Logits:", logits)
        #print("Targets:", targets)
        return compute_loss(logits, targets)

    # Compute the loss and gradients
    loss, grads = value_and_grad(loss_fn)(state.params)

    grads = jax.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)


    state = state.apply_gradients(grads=grads)

    # Split the RNG key for the next step
    _, next_rng_key = random.split(rng_key)

    return state, loss, next_rng_key

#@jit
def eval_step(params, inputs, targets):
    print("Doing eval...")
    logits = model.apply({'params': params}, inputs, training=False)
    return compute_loss(logits, targets)

# Training loop
num_epochs = 1

for epoch in range(num_epochs):
    print(f"Starting epoch {epoch+1}...")
    train_losses, val_losses = [], []
    rng_key = random.PRNGKey(epoch)

    # Training phase
    for inputs, targets in train_loader:
        inputs, targets = jnp.array(inputs), jnp.array(targets)
        state, loss, rng_key = train_step(state, inputs, targets, rng_key)

        # Check for NaNs in the model parameters
        nan_params = tree_util.tree_map(lambda x: jnp.any(jnp.isnan(x)), state.params)
        #print("NaN parameters:", nan_params)

        train_losses.append(loss)
        print(f"Train loss: {loss}")
        logger.log_train_loss(loss)

    # Save the model parameters after each epoch
    epoch_dir = os.path.join(model_dir, f'epoch_{epoch+1}_{datetime.now().strftime("%Y%m%d-%H%M%S")}')
    os.makedirs(epoch_dir, exist_ok=True)
    flat_params, _ = tree_util.tree_flatten(state.params)
    for i, param in enumerate(flat_params):
        param_path = os.path.join(epoch_dir, f'param_{i}.npy')
        np.save(param_path, param)
        print(f"Saved parameter {i} to {param_path}")

    # Compute mean training loss
    mean_train_loss = jnp.mean(jnp.stack(train_losses))
    print(f"Mean train loss: {mean_train_loss:.4f}")

    # Validation phase
    for inputs, targets in val_loader:
        inputs, targets = jnp.array(inputs), jnp.array(targets)
        loss = eval_step(state.params, inputs, targets)
        val_losses.append(loss)
        print(f"Validation loss: {loss}")
        logger.log_val_loss(loss)

    # Compute mean validation loss
    mean_val_loss = jnp.mean(jnp.stack(val_losses))
    print(f"Epoch {epoch+1}: Train Loss: {mean_train_loss:.4f}, Val Loss: {mean_val_loss:.4f}")