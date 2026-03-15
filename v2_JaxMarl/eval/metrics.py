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

    # Transition batch shapes: (seq_len, num_envs, ...) from loop.py
    loc = transition_batch.local_obs
    prop = transition_batch.proprioception
    msg_true = transition_batch.message

    # True forward pass
    _, true_logits = jax.lax.scan(scan_fn, init_doer_carry, (msg_true, loc, prop))
    
    # Ablated forward pass: shuffle messages independently over time for each env.
    env_keys = jax.random.split(rng, msg_true.shape[1])
    msg_shuffled = jax.vmap(
        lambda key, msgs: jax.random.permutation(key, msgs, axis=0)
    )(env_keys, jnp.swapaxes(msg_true, 0, 1))
    msg_shuffled = jnp.swapaxes(msg_shuffled, 0, 1)
    
    _, ablated_logits = jax.lax.scan(scan_fn, init_doer_carry, (msg_shuffled, loc, prop))
    
    # Calculate CIC: divergence in deterministic policy
    true_actions = jnp.argmax(true_logits, axis=-1)
    ablated_actions = jnp.argmax(ablated_logits, axis=-1)
    
    cic = jnp.mean((true_actions != ablated_actions).astype(jnp.float32))
    
    return cic
