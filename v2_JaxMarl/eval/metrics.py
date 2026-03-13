import jax
import jax.numpy as jnp
from typing import Tuple, Callable
import functools

@functools.partial(jax.jit, static_argnames=("doer_apply_fn",))
def compute_cic(
    doer_apply_fn: Callable,
    doer_params: dict,
    transition_batch, 
    init_doer_carry: Tuple[jnp.ndarray, jnp.ndarray],
    rng: jax.random.PRNGKey
) -> jnp.ndarray:
    """
    Computes Causal Influence of Communication (CIC) by ablating the Seer's message
    and measuring the divergence in the Doer's resulting deterministic actions.
    """
    
    def scan_fn(carry, step_data):
        doer_carry = carry
        msg, loc, prop = step_data
        
        next_doer_carry, logits = doer_apply_fn(
            {"params": doer_params},
            doer_carry,
            loc,
            prop,
            msg
        )
        return next_doer_carry, logits

    # Transition batch shapes: (seq_len, ...) from loop.py
    # We add batch dim of 1 for the network inputs
    loc = transition_batch.local_obs[:, None, ...]
    prop = transition_batch.proprioception[:, None, ...]
    msg_true = transition_batch.message[:, None, ...]

    # True forward pass
    _, true_logits = jax.lax.scan(scan_fn, init_doer_carry, (msg_true, loc, prop))
    
    # Ablated forward pass (shuffle messages over time)
    msg_shuffled = jax.random.permutation(rng, msg_true, independent=True)
    
    _, ablated_logits = jax.lax.scan(scan_fn, init_doer_carry, (msg_shuffled, loc, prop))
    
    # Calculate CIC: divergence in deterministic policy
    true_actions = jnp.argmax(true_logits, axis=-1)
    ablated_actions = jnp.argmax(ablated_logits, axis=-1)
    
    cic = jnp.mean((true_actions != ablated_actions).astype(jnp.float32))
    
    return cic