import jax.numpy as jnp


def mask_pick_actions_until_menu_visible(logits: jnp.ndarray, menu_images: jnp.ndarray) -> jnp.ndarray:
    """Disable pick actions until the menu is visible for that agent."""
    menu_feature_axes = tuple(range(logits.ndim - 1, menu_images.ndim))
    menu_visible = jnp.any(menu_images > 0.0, axis=menu_feature_axes)
    pick_mask = jnp.arange(logits.shape[-1]) >= 5
    invalid_pick_mask = jnp.logical_and(~menu_visible[..., None], pick_mask)
    large_negative = jnp.asarray(-1.0e9, dtype=logits.dtype)
    return jnp.where(invalid_pick_mask, large_negative, logits)
