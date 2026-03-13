import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
import chex
from typing import Tuple, Any, Callable
import distrax 

# 1. Data Structures: Meticulous shape tracking
# Using chex helps enforce accuracy by allowing us to assert shapes later.
@chex.dataclass
class Transition:
    global_obs: chex.Array  # For the Critic (CTDE)
    local_obs: chex.Array   # For the Actor
    action: chex.Array
    log_prob: chex.Array
    value: chex.Array
    reward: chex.Array
    done: chex.Array
    advantage: chex.Array
    return_val: chex.Array

# 2. Loss Functions: Separated from network definitions
def calculate_actor_loss(
    actor_apply_fn: Callable,
    actor_params: Any,
    transition_batch: Transition,
    clip_eps: float = 0.2,
    entropy_coef: float = 0.01
) -> Tuple[jnp.ndarray, dict]:
    """Calculates the PPO clipped surrogate loss using distrax."""
    
    # Get logits from the network (e.g., the Seer's message head)
    logits = actor_apply_fn({"params": actor_params}, transition_batch.local_obs)
    
    # Create a distribution object
    # For discrete communication/actions, Categorical is the standard
    pi = distrax.Categorical(logits=logits)
    
    # Use the distribution to get new log_probs and entropy
    # transition_batch.action contains the indices of the symbols sent
    new_log_probs = pi.log_prob(transition_batch.action)
    entropy = pi.entropy().mean()
    
    # Calculate the ratio (pi_new / pi_old)
    logratio = new_log_probs - transition_batch.log_prob
    ratio = jnp.exp(logratio)
    
    # Advantage normalization
    adv = transition_batch.advantage
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    
    # Standard PPO Clipped Objective
    loss_unclipped = ratio * adv
    loss_clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    
    actor_loss = -jnp.minimum(loss_unclipped, loss_clipped).mean()
    entropy_loss = entropy_coef * entropy
    
    total_actor_loss = actor_loss - entropy_loss
    
    return total_actor_loss, {
        "actor_loss": actor_loss, 
        "entropy": entropy,
        "ratio": ratio.mean()
    }

def calculate_critic_loss(
    critic_apply_fn: Callable,
    critic_params: Any,
    transition_batch: Transition,
    value_clip: float = 0.2
) -> Tuple[jnp.ndarray, dict]:
    """Calculates the value loss for the centralized critic."""
    
    # The critic uses the global observation (CTDE)
    values = critic_apply_fn({"params": critic_params}, transition_batch.global_obs)
    
    value_pred_clipped = transition_batch.value + jnp.clip(
        values - transition_batch.value, -value_clip, value_clip
    )
    
    value_losses = jnp.square(values - transition_batch.return_val)
    value_losses_clipped = jnp.square(value_pred_clipped - transition_batch.return_val)
    
    critic_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
    
    return critic_loss, {"critic_loss": critic_loss}

# 3. The Update Step: JIT-compiled gradient application
@jax.jit
def update_actor(
    actor_state: TrainState, 
    transition_batch: Transition
) -> Tuple[TrainState, dict]:
    """Computes gradients and updates the actor network."""
    
    # jax.value_and_grad returns both the loss value and the gradients
    loss_fn = lambda params: calculate_actor_loss(
        actor_state.apply_fn, params, transition_batch
    )
    
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(actor_state.params)
    
    # Apply gradients using the optimizer stored in the TrainState
    new_actor_state = actor_state.apply_gradients(grads=grads)
    return new_actor_state, metrics

@jax.jit
def update_critic(
    critic_state: TrainState, 
    transition_batch: Transition
) -> Tuple[TrainState, dict]:
    """Computes gradients and updates the critic network."""
    
    loss_fn = lambda params: calculate_critic_loss(
        critic_state.apply_fn, params, transition_batch
    )
    
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(critic_state.params)
    
    new_critic_state = critic_state.apply_gradients(grads=grads)
    return new_critic_state, metrics