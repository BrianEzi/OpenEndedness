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
        msg, loc, prop, menu = step_data
        
        next_doer_carry, logits = doer_apply_fn(
            {"params": doer_params},
            doer_carry,
            loc,
            prop,
            msg,
            menu
        )
        return next_doer_carry, logits

    # Transition batch shapes: (seq_len, num_envs, ...) from loop.py
    loc = transition_batch.local_obs
    prop = transition_batch.proprioception
    msg_true = transition_batch.message
    menu = transition_batch.menu_images

    # True forward pass
    _, true_logits = jax.lax.scan(scan_fn, init_doer_carry, (msg_true, loc, prop, menu))
    
    # Ablated forward pass: shuffle messages independently over time for each env.
    env_keys = jax.random.split(rng, msg_true.shape[1])
    msg_shuffled = jax.vmap(
        lambda key, msgs: jax.random.permutation(key, msgs, axis=0)
    )(env_keys, jnp.swapaxes(msg_true, 0, 1))
    msg_shuffled = jnp.swapaxes(msg_shuffled, 0, 1)
    
    _, ablated_logits = jax.lax.scan(scan_fn, init_doer_carry, (msg_shuffled, loc, prop, menu))
    
    # Calculate CIC: divergence in deterministic policy
    true_actions = jnp.argmax(true_logits, axis=-1)
    ablated_actions = jnp.argmax(ablated_logits, axis=-1)
    
    cic = jnp.mean((true_actions != ablated_actions).astype(jnp.float32))
    
    return cic


@functools.partial(jax.jit, static_argnames=("doer_apply_fn",))
def compute_two_doer_cic(
    doer_apply_fn: Callable,
    doer_params: dict,
    transition_batch,
    init_doer_carry: Tuple[jnp.ndarray, jnp.ndarray],
    rng: jax.random.PRNGKey,
) -> jnp.ndarray:
    """
    Computes CIC for the two-Doer setting by ablating each private message stream
    and measuring how often the shared Doer policy changes its greedy action.
    """

    def scan_fn(carry, step_data):
        doer_carry = carry
        msg, loc, prop, menu = step_data
        batch_size, num_doers = loc.shape[:2]
        flat_loc = loc.reshape((batch_size * num_doers,) + loc.shape[2:])
        flat_prop = prop.reshape((batch_size * num_doers,) + prop.shape[2:])
        flat_msg = msg.reshape((batch_size * num_doers,) + msg.shape[2:])
        flat_menu = menu.reshape((batch_size * num_doers,) + menu.shape[2:])
        flat_carry = jax.tree_util.tree_map(
            lambda x: x.reshape((batch_size * num_doers,) + x.shape[2:]),
            doer_carry,
        )
        next_flat_carry, flat_logits = doer_apply_fn(
            {"params": doer_params},
            flat_carry,
            flat_loc,
            flat_prop,
            flat_msg,
            flat_menu,
        )
        next_carry = jax.tree_util.tree_map(
            lambda x: x.reshape((batch_size, num_doers) + x.shape[1:]),
            next_flat_carry,
        )
        logits = flat_logits.reshape((batch_size, num_doers, flat_logits.shape[-1]))
        return next_carry, logits

    loc = transition_batch.local_obs
    prop = transition_batch.proprioception
    msg_true = transition_batch.message
    menu = transition_batch.menu_images

    _, true_logits = jax.lax.scan(scan_fn, init_doer_carry, (msg_true, loc, prop, menu))

    flat_keys = jax.random.split(rng, msg_true.shape[1] * msg_true.shape[2])
    msg_shuffled = jnp.swapaxes(msg_true, 0, 1)
    msg_shuffled = jnp.swapaxes(msg_shuffled, 1, 2)
    msg_shuffled = msg_shuffled.reshape((-1, msg_true.shape[0], msg_true.shape[-1]))
    msg_shuffled = jax.vmap(
        lambda key, msgs: jax.random.permutation(key, msgs, axis=0)
    )(flat_keys, msg_shuffled)
    msg_shuffled = msg_shuffled.reshape(
        (msg_true.shape[1], msg_true.shape[2], msg_true.shape[0], msg_true.shape[-1])
    )
    msg_shuffled = jnp.swapaxes(msg_shuffled, 1, 2)
    msg_shuffled = jnp.swapaxes(msg_shuffled, 0, 1)

    _, ablated_logits = jax.lax.scan(scan_fn, init_doer_carry, (msg_shuffled, loc, prop, menu))

    true_actions = jnp.argmax(true_logits, axis=-1)
    ablated_actions = jnp.argmax(ablated_logits, axis=-1)
    cic = jnp.mean((true_actions != ablated_actions).astype(jnp.float32))
    return cic
