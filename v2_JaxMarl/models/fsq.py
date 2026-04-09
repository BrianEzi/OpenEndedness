import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence

class FSQ(nn.Module):
    """
    Finite Scalar Quantization (FSQ) module.
    Projects a continuous vector into a discrete hypercube.
    """
    # A sequence of integers defining the number of levels (L) per dimension (d).
    # e.g., levels=[5, 5, 5] means d=3 dimensions, each with 5 discrete levels.
    levels: Sequence[int]

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            z: The continuous "thought vector" from the Seer's reasoning module.
               Expected shape: (batch_size, ..., d) where d == len(self.levels)
        Returns:
            z_ste: The quantized vector with gradients preserved via STE.
        """
        levels = jnp.asarray(self.levels, dtype=z.dtype)
        
        # 1. Bounding: Restrict the unbounded input z to the range [-1, 1].
        # We use tanh as a standard, proven method for this projection.
        z_bound = jnp.tanh(z)
        
        # 2. Scaling: Map the [-1, 1] range to the grid [0, L - 1].
        # E.g., for L=5, this maps [-1, 1] to [0, 4].
        half_width = (levels - 1) / 2.0
        z_scaled = z_bound * half_width + half_width
        
        # 3. Quantization: Snap to the nearest integer grid point.
        if self.has_rng('noise'):
            # Inject uniform noise during training to "shake" the FSQ and prevent early mode collapse
            noise = jax.random.uniform(self.make_rng('noise'), z_scaled.shape, minval=-0.2, maxval=0.2)
            z_quantized = jnp.round(z_scaled + noise)
        else:
            z_quantized = jnp.round(z_scaled)
        
        # 4. Straight-Through Estimator (STE) Trick in JAX
        # Forward pass: Evaluates to z_quantized.
        # Backward pass: jax.lax.stop_gradient blocks the gradient from the 
        # non-differentiable z_quantized, so the gradient flows directly through z_scaled.
        z_ste = z_scaled + jax.lax.stop_gradient(z_quantized - z_scaled)
        
        return z_ste