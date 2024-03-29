import numpy as np
import os
import pickle
import glob
from jax import numpy as jnp
from jax import tree_util, random
from flax import linen as nn
from flax.training import train_state
import optax

input_file_path = "/Users/scott/Downloads/Jax/math_problems_dataset.txt"
output_dir = "/Users/scott/Downloads/Jax"

with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

chars = sorted(list(set(data)))
stoi = { ch:i for i,ch in enumerate(chars) }

output_dir = "/Users/scott/Downloads/Jax"
meta_path = os.path.join(output_dir, 'meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
itos = meta['itos']
stoi = meta['stoi']
vocab_size = meta['vocab_size']

# Positional Encoding
class PositionalEncoding(nn.Module):
    vocab_size: int
    d_model: int
    max_len: int = 5000

    def setup(self):
        position = np.arange(0, self.max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        pe = np.zeros((self.max_len, self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = jnp.array(pe)

    def __call__(self, x):
        return x + self.pe[None, :x.shape[1], :]

# Transformer Block
class TransformerBlock(nn.Module):
    d_model: int
    num_heads: int
    d_ff: int
    dropout_rate: float = 0

    @nn.compact
    def __call__(self, x, training):
        attn_out = nn.SelfAttention(num_heads=self.num_heads, qkv_features=self.d_model, dropout_rate=self.dropout_rate, deterministic=not training)(x)
        attn_out = nn.LayerNorm()(attn_out + x)
        ff_out = nn.Dense(self.d_ff)(attn_out)
        ff_out = nn.relu(ff_out)
        ff_out = nn.Dense(self.d_model)(ff_out)
        ff_out = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(ff_out)
        return nn.LayerNorm()(ff_out + attn_out)

# Transformer Model
class Transformer(nn.Module):
    vocab_size: int
    d_model: int = 32 #neuron per layer
    num_heads: int = 4
    num_layers: int = 6
    d_ff: int = 128 # feed forward layer

    def setup(self):
        self.token_embedding = nn.Embed(self.vocab_size, self.d_model)
        self.pos_encoding = PositionalEncoding(self.vocab_size, self.d_model)
        self.transformer_blocks = [TransformerBlock(self.d_model, self.num_heads, self.d_ff) for _ in range(self.num_layers)]
        self.dense_output = nn.Dense(self.vocab_size)

    def __call__(self, input_ids, training=True):
        x = self.token_embedding(input_ids)
        x = self.pos_encoding(x)
        for block in self.transformer_blocks:
            x = block(x, training)
        return self.dense_output(x)

def load_model(model_dir='saved_models'):
    # Get the list of all directories in the model_dir
    dirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]

    # Find the most recently modified directory
    most_recent_dir = max(dirs, key=lambda x: os.path.getmtime(os.path.join(model_dir, x)))
    latest_batch_dir = os.path.join(model_dir, most_recent_dir)
    print(f"Most recent batch directory: {latest_batch_dir}")

    # Get all .npy files in the most recent directory
    all_files = glob.glob(os.path.join(latest_batch_dir, 'param_*.npy'))
    all_files = [f for f in all_files if os.path.isfile(f)]

    model = Transformer(vocab_size=vocab_size)
    rng = random.PRNGKey(0)
    dummy_input = jnp.ones((1, 25), dtype=jnp.int32)
    initial_structure = model.init(rng, dummy_input)

    loaded_params = {}
    for param_file in all_files:
        param_name = os.path.basename(param_file).split('_')[1].split('.')[0]
        param_value = np.load(param_file)
        loaded_params[param_name] = param_value

    _, treedef = tree_util.tree_flatten(initial_structure['params'])
    #print(f"Keys in loaded_params: {list(loaded_params.keys())}")
    unflat_params = treedef.unflatten([loaded_params[str(i)] for i in range(len(loaded_params))])
    #print("Parameters successfully reconstructed.")

    tx = optax.adam(learning_rate=1e-4)
    state = train_state.TrainState.create(apply_fn=model.apply, params=unflat_params, tx=tx)
    print("Model loading complete.")
    return state


def preprocess_inference_input(query, stoi, input_length=9, pad_token=17, equals_token=None):
    """
    Prepares and encodes the query for model input, ensuring it is exactly 9 characters long,
    including padding and an equals sign at the end.
    """
    if equals_token is None:
        equals_token = stoi['=']
    encoded_query = [stoi.get(char, pad_token) for char in query] + [equals_token]
    padded_query = encoded_query[:input_length-1] + [pad_token] * max(0, input_length - 1 - len(encoded_query)) + [equals_token]
    return np.array(padded_query, dtype=np.int32).reshape(1, -1)

def model_inference(input_ids, model_state):
    """
    Executes model inference given the preprocessed input IDs.
    """
    logits = model_state.apply_fn({'params': model_state.params}, input_ids, training=False)
    return logits

def decode_model_output(logits, itos, pad_token=-1):
    """
    Decodes the logits from the model output to a human-readable string,
    ignoring any padding tokens.
    """
    predicted_indices = jnp.argmax(logits, axis=-1)[0]
    decoded_output = ''.join(itos.get(int(idx), '') for idx in predicted_indices if idx != pad_token)
    return decoded_output


if __name__ == "__main__":
    model_state = load_model(model_dir=os.path.join(output_dir, 'saved_models'))

    query = "1*1"
    input_ids = preprocess_inference_input(query, stoi)
    logits = model_inference(input_ids, model_state)
    prediction = decode_model_output(logits, itos)

    print(f"Raw input tokens: {input_ids.flatten().tolist()}")
    print(f"Raw output tokens: {jnp.argmax(logits, axis=-1)[0].tolist()}")
    print(f"Query: '{query}'")
    print(f"Prediction: '{prediction}'")