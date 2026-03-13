import jax
import jax.numpy as jnp
import flax.linen as nn


class SeerCNN(nn.Module):
    """
    CNN encoder for the Seer agent.
    Takes a (4, 5, 11) spatial tensor and outputs a 128-dim embedding.
    Same architecture as the Cook CNN (Foerster et al. 2016, Sukhbaatar et al. 2016).
    """

    @nn.compact
    def __call__(self, x):
        # x: (4, 5, 11)
        x = nn.Conv(features=32, kernel_size=(5, 5))(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = x.reshape(-1)
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)

        return x  # (128,)


class SeerEncoder(nn.Module):
    """
    Full Seer encoder: CNN + LayerNorm + GRU.
    Takes a (4, 5, 11) spatial tensor and a hidden state.
    Returns the updated hidden state and GRU output, both (128,).
    (Foerster et al. 2016, Kim et al. 2019)
    """

    @nn.compact
    def __call__(self, hidden, x):
        embedding = SeerCNN()(x)              # (128,)
        embedding = nn.LayerNorm()(embedding) # stabilise before GRU
        new_hidden, output = nn.GRUCell(128)(hidden, embedding)
        return new_hidden, output  # both (128,)

    @staticmethod
    def initialize_hidden(rng):
        """Create a blank hidden state (all zeros) for start of episode."""
        return nn.GRUCell(128).initialize_carry(rng, (128,))