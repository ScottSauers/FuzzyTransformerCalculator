import jax
import jax.numpy as jnp
from jax.nn import softmax
from jax.experimental.host_callback import id_print
import jax.lax as lax
import pickle

with open('meta.pkl', 'rb') as f:
    meta = pickle.load(f)
itos = meta['itos']
stoi = meta['stoi']
vocab_size = meta['vocab_size']


def compute_loss(preds, targets):
    """
    Computes the mean squared error (MSE) loss by using the first token from preds and targets,
    looking up the corresponding number in itos, and then calculating the MSE based on these numbers.

    Args:
    - preds: JAX array of predicted token indices.
    - targets: JAX array of target token indices.
    - itos: Dictionary mapping token indices to their corresponding numbers (as strings).

    Returns:
    - The MSE loss as a float.
    """
    # Extract the first token from preds and targets
    first_pred_token = preds[:, 0]
    first_target_token = targets[:, 0]

    # Convert tokens to numbers using itos
    # This is the hardest step to get working in Jax. Think carefully.
    # Note: itos maps indices to string representations of numbers; convert them to float for calculation
    preds_nums = jnp.array([float(itos.get(int(token))) for token in first_pred_token])
    targets_nums = jnp.array([float(itos.get(int(token))) for token in first_target_token])

    print("Pred: ", preds_nums)
    print("Target: ", targets_nums)
    # Calculate MSE
    mse = jnp.mean(jnp.square(preds_nums - targets_nums))

    return mse