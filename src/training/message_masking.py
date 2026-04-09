import jax.numpy as jnp


def hard_mask_inactive_message_bits(messages: jnp.ndarray, active_bits: int) -> jnp.ndarray:
    """Force trailing message dimensions to zero while keeping the architecture fixed."""
    bit_mask = (jnp.arange(messages.shape[-1]) < active_bits).astype(messages.dtype)
    return messages * bit_mask
