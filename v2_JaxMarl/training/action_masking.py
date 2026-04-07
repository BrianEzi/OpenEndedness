import jax.numpy as jnp


def mask_pick_actions_until_menu_visible(
    logits: jnp.ndarray,
    menu_images: jnp.ndarray,
    pick_only_phase: bool = False,
) -> jnp.ndarray:
    """Disable invalid picks and optionally freeze navigation during the pick-only phase."""
    menu_feature_axes = tuple(range(logits.ndim - 1, menu_images.ndim))
    menu_visible = jnp.any(menu_images > 0.0, axis=menu_feature_axes)
    action_ids = jnp.arange(logits.shape[-1])
    pick_mask = action_ids >= 5
    invalid_pick_mask = jnp.logical_and(~menu_visible[..., None], pick_mask)
    large_negative = jnp.asarray(-1.0e9, dtype=logits.dtype)
    masked_logits = jnp.where(invalid_pick_mask, large_negative, logits)
    if pick_only_phase:
        navigation_mask = jnp.logical_and(action_ids >= 1, action_ids < 5)
        masked_logits = jnp.where(navigation_mask, large_negative, masked_logits)
    return masked_logits
