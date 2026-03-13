import jax
import jax.numpy as jnp
import flax.linen as nn


class SeerFSQ(nn.Module):
    """
    Finite Scalar Quantization head for the Seer agent.
    Projects the GRU output (128,) to n_slots discrete tokens.

    Each slot is quantised to {-1, 0, +1} via tanh + round.
    A straight-through estimator lets gradients flow through the rounding step.
    The integer index (base n_levels encoding) is returned for logging and TopSim.

    (Mentzer et al. 2023, ClusterComm 2024)
    """

    n_slots:  int = 8  # number of quantised dimensions
    n_levels: int = 3  # values per slot: {-1, 0, +1}

    @nn.compact
    def __call__(self, z):
        # z: (128,) GRU output
        z = nn.Dense(self.n_slots)(z)   # project to n_slots
        z = nn.tanh(z)                  # bound to (-1, +1)

        half      = self.n_levels // 2
        z_scaled  = z * half
        z_rounded = jnp.round(z_scaled)

        # straight-through: discrete forward, continuous backward
        z_q = z_scaled + jax.lax.stop_gradient(z_rounded - z_scaled)

        # base-n_levels encoding → single integer for logging and TopSim
        shifted = (z_rounded + half).astype(jnp.int32)
        powers  = jnp.array(
            [self.n_levels ** i for i in range(self.n_slots)],
            dtype=jnp.int32,
        )
        index = jnp.sum(shifted * powers)

        return z_q, index  # (n_slots,), scalar
