import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
import chex
from typing import Tuple, Any, Callable
import distrax 
import optax
from training.action_masking import mask_pick_actions_until_menu_visible

# 1. Data Structures: Meticulous shape tracking
# Using chex helps enforce accuracy by allowing us to assert shapes later.
@chex.dataclass
class Transition:
    global_obs: chex.Array  # For the Critic (CTDE)
    symbolic_obs: chex.Array # For the Seer
    local_obs: chex.Array   # For the Doer
    proprioception: chex.Array # For the Doer
    message: chex.Array     # For CIC and Heatmap logging
    target_images: chex.Array
    menu_images: chex.Array
    doer_action: chex.Array
    doer_log_prob: chex.Array
    seer_action: chex.Array
    seer_log_prob: chex.Array
    value: chex.Array
    reward: chex.Array
    task_reward: chex.Array
    progress_reward: chex.Array
    follow_reward: chex.Array
    cic_reward_component: chex.Array
    cic_score: chex.Array
    step_penalty_component: chex.Array
    bump_penalty_component: chex.Array
    done: chex.Array
    advantage: chex.Array
    return_val: chex.Array


@chex.dataclass
class TwoDoerTransition:
    global_obs: chex.Array
    symbolic_obs: chex.Array
    local_obs: chex.Array
    proprioception: chex.Array
    message: chex.Array
    target_images: chex.Array
    menu_images: chex.Array
    doer_action: chex.Array
    doer_log_prob: chex.Array
    value: chex.Array
    reward: chex.Array
    task_reward: chex.Array
    individual_selection_reward: chex.Array
    valid_selection_count: chex.Array
    correct_selection_count: chex.Array
    progress_reward_per_doer: chex.Array
    step_penalty_component: chex.Array
    wall_penalty_component: chex.Array
    collision_penalty_component: chex.Array
    done: chex.Array
    advantage: chex.Array
    return_val: chex.Array


def _build_message_codebook(message_levels: Tuple[int, ...], dtype: jnp.dtype) -> jnp.ndarray:
    axes = [jnp.arange(level, dtype=dtype) for level in message_levels]
    mesh = jnp.meshgrid(*axes, indexing="ij")
    return jnp.stack(mesh, axis=-1).reshape((-1, len(message_levels)))


def _compute_message_entropy_metrics(
    discrete_messages: chex.Array,
    message_levels: Tuple[int, ...],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Approximate discrete-code usage with a soft histogram so gradients can flow."""
    flat_messages = discrete_messages.reshape((-1, discrete_messages.shape[-1]))
    codebook = _build_message_codebook(message_levels, flat_messages.dtype)
    sq_distances = jnp.sum(
        jnp.square(flat_messages[:, None, :] - codebook[None, :, :]),
        axis=-1,
    )
    assignment_probs = nn.softmax(-10.0 * sq_distances, axis=-1)
    code_probs = assignment_probs.mean(axis=0)
    code_probs = code_probs / (code_probs.sum() + 1e-8)
    message_entropy = -jnp.sum(code_probs * jnp.log(code_probs + 1e-8))

    num_codes = codebook.shape[0]
    if num_codes > 1:
        normalized_entropy = message_entropy / jnp.log(jnp.asarray(num_codes, dtype=flat_messages.dtype))
    else:
        normalized_entropy = jnp.asarray(0.0, dtype=flat_messages.dtype)

    dominant_code_prob = jnp.max(code_probs)
    return message_entropy, normalized_entropy, dominant_code_prob

# 2. Loss Functions: Separated from network definitions
def calculate_actor_losses(
    seer_apply_fn: Callable,
    doer_apply_fn: Callable,
    actor_params: dict, # Changed from Any to dict
    transition_batch: Transition,
    init_seer_carry: Tuple[jnp.ndarray, jnp.ndarray], # Changed from Any to Tuple
    init_doer_carry: Tuple[jnp.ndarray, jnp.ndarray], # Changed from Any to Tuple
    control_mode: jnp.ndarray,
    message_levels: Tuple[int, ...],
    clip_eps: float = 0.2,
    entropy_coef: float = 0.01,
    seer_entropy_coef: jnp.ndarray = jnp.array(0.01)
) -> Tuple[jnp.ndarray, dict]:
    """Calculates separate PPO losses for the Seer and Doer reward streams."""
    
    # We must assert shape to prevent silent broadcasting bugs.
    # transition_batch has shape (batch_size, sequence_length, ...) or just (sequence_length, ...)
    # Wait, loop.py returns (num_steps, ...) for single env. Let's assume unbatched or reshape-able.
    # If the user is unrolling over dimension 0:
    
    def scan_fn(carry, transition_step):
        # Step shape assertions:
        # chex.assert_rank(transition_step.action, 1) # (batch_size,)
        # chex.assert_rank(transition_step.global_obs, 4) # (batch_size, H, W, C)
        
        seer_carry, doer_carry = carry
        
        # Seer Forward Pass
        next_seer_carry, discrete_message, thought_vector, seer_nav_logits = seer_apply_fn(
            {"params": actor_params["seer"]},
            seer_carry,
            transition_step.global_obs,
            transition_step.symbolic_obs,
            transition_step.target_images,
        )
        
        # Doer Forward Pass
        next_doer_carry, logits = doer_apply_fn(
            {"params": actor_params["doer"]},
            doer_carry,
            transition_step.local_obs,
            transition_step.proprioception,
            discrete_message,
            transition_step.menu_images
        )
        return (next_seer_carry, next_doer_carry), (
            logits,
            seer_nav_logits,
            discrete_message,
            thought_vector,
        )
        
    _, (doer_logits, seer_nav_logits, discrete_messages, thought_vectors) = jax.lax.scan(
        scan_fn, 
        (init_seer_carry, init_doer_carry), 
        transition_batch
    )

    communication_mode = control_mode == 1
    doer_pi = distrax.Categorical(logits=doer_logits)
    seer_pi = distrax.Categorical(logits=seer_nav_logits)
    doer_new_log_probs = doer_pi.log_prob(transition_batch.doer_action)
    seer_nav_new_log_probs = seer_pi.log_prob(transition_batch.seer_action)
    doer_entropy = doer_pi.entropy().mean()
    seer_nav_entropy = seer_pi.entropy().mean()

    seer_adv = transition_batch.advantage[..., 0]
    doer_adv = transition_batch.advantage[..., 1]
    seer_adv = (seer_adv - seer_adv.mean()) / (seer_adv.std() + 1e-8)
    doer_adv = (doer_adv - doer_adv.mean()) / (doer_adv.std() + 1e-8)

    seer_old_log_probs = jnp.where(
        communication_mode,
        transition_batch.doer_log_prob,
        transition_batch.seer_log_prob,
    )
    seer_new_log_probs = jnp.where(
        communication_mode,
        doer_new_log_probs,
        seer_nav_new_log_probs,
    )
    seer_logratio = seer_new_log_probs - seer_old_log_probs
    seer_ratio = jnp.exp(seer_logratio)

    seer_loss_unclipped = seer_ratio * seer_adv
    seer_loss_clipped = jnp.clip(seer_ratio, 1.0 - clip_eps, 1.0 + clip_eps) * seer_adv
    seer_actor_loss = -jnp.minimum(seer_loss_unclipped, seer_loss_clipped).mean()

    doer_logratio = doer_new_log_probs - transition_batch.doer_log_prob
    doer_ratio = jnp.exp(doer_logratio)
    doer_loss_unclipped = doer_ratio * doer_adv
    doer_loss_clipped = jnp.clip(doer_ratio, 1.0 - clip_eps, 1.0 + clip_eps) * doer_adv
    doer_actor_loss = -jnp.minimum(doer_loss_unclipped, doer_loss_clipped).mean()
    doer_actor_loss = jnp.where(communication_mode, doer_actor_loss, 0.0)

    message_entropy, message_entropy_normalized, dominant_code_prob = (
        _compute_message_entropy_metrics(discrete_messages, message_levels)
    )
    seer_bonus = jnp.where(communication_mode, message_entropy, seer_nav_entropy)

    seer_loss = seer_actor_loss - seer_entropy_coef * seer_bonus
    doer_loss = doer_actor_loss - jnp.where(communication_mode, entropy_coef * doer_entropy, 0.0)

    return (seer_loss, doer_loss), {
        "seer_actor_loss": seer_actor_loss,
        "doer_actor_loss": doer_actor_loss,
        "entropy": jnp.where(communication_mode, doer_entropy, seer_nav_entropy),
        "seer_ratio": seer_ratio.mean(),
        "doer_ratio": doer_ratio.mean(),
        "message_entropy": message_entropy,
        "message_entropy_normalized": message_entropy_normalized,
        "message_dominant_probability": dominant_code_prob,
        "seer_nav_entropy": seer_nav_entropy,
        "discrete_messages": discrete_messages
    }

def calculate_critic_loss(
    critic_apply_fn: Callable,
    critic_params: Any,
    transition_batch: Transition,
    value_clip: float = 0.2
) -> Tuple[jnp.ndarray, dict]:
    """Calculates the value loss for the centralized critic."""
    
    # Assert shape
    chex.assert_rank(transition_batch.global_obs, 4) # (batch_size, H, W, C) since it is flattened over sequence
    
    # The critic uses the global observation (CTDE)
    values = critic_apply_fn({"params": critic_params}, transition_batch.global_obs)

    value_pred_clipped = transition_batch.value + jnp.clip(
        values - transition_batch.value, -value_clip, value_clip
    )
    
    value_losses = jnp.square(values - transition_batch.return_val)
    value_losses_clipped = jnp.square(value_pred_clipped - transition_batch.return_val)
    
    critic_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
    
    return critic_loss, {"critic_loss": critic_loss}


def calculate_two_doer_actor_losses(
    seer_apply_fn: Callable,
    doer_apply_fn: Callable,
    actor_params: dict,
    transition_batch: TwoDoerTransition,
    init_seer_carry: Tuple[jnp.ndarray, jnp.ndarray],
    init_doer_carry: Tuple[jnp.ndarray, jnp.ndarray],
    message_levels: Tuple[int, ...],
    clip_eps: float = 0.2,
    entropy_coef: float = 0.01,
    seer_entropy_coef: jnp.ndarray = jnp.array(0.01),
) -> Tuple[jnp.ndarray, dict]:
    """PPO actor losses for one Seer coordinating two embodied Doers."""

    def scan_fn(carry, transition_step):
        seer_carry, doer_carry = carry
        next_seer_carry, discrete_messages, _, _ = seer_apply_fn(
            {"params": actor_params["seer"]},
            seer_carry,
            transition_step.global_obs,
            transition_step.symbolic_obs,
            transition_step.target_images,
        )
        batch_size, num_doers = transition_step.local_obs.shape[:2]
        flat_local_obs = transition_step.local_obs.reshape(
            (batch_size * num_doers,) + transition_step.local_obs.shape[2:]
        )
        flat_proprioception = transition_step.proprioception.reshape(
            (batch_size * num_doers,) + transition_step.proprioception.shape[2:]
        )
        flat_messages = discrete_messages.reshape(
            (batch_size * num_doers,) + discrete_messages.shape[2:]
        )
        flat_doer_carry = jax.tree_util.tree_map(
            lambda x: x.reshape((batch_size * num_doers,) + x.shape[2:]),
            doer_carry,
        )
        next_flat_doer_carry, flat_logits = doer_apply_fn(
            {"params": actor_params["doer"]},
            flat_doer_carry,
            flat_local_obs,
            flat_proprioception,
            flat_messages,
            transition_step.menu_images.reshape((batch_size * num_doers,) + transition_step.menu_images.shape[2:])
        )
        next_doer_carry = jax.tree_util.tree_map(
            lambda x: x.reshape((batch_size, num_doers) + x.shape[1:]),
            next_flat_doer_carry,
        )
        doer_logits = flat_logits.reshape((batch_size, num_doers, flat_logits.shape[-1]))
        doer_logits = mask_pick_actions_until_menu_visible(
            doer_logits,
            transition_step.menu_images,
        )
        return (next_seer_carry, next_doer_carry), (doer_logits, discrete_messages)

    _, (doer_logits, discrete_messages) = jax.lax.scan(
        scan_fn,
        (init_seer_carry, init_doer_carry),
        transition_batch,
    )

    doer_pi = distrax.Categorical(logits=doer_logits)
    doer_new_log_probs = doer_pi.log_prob(transition_batch.doer_action)
    doer_entropy = doer_pi.entropy().mean()

    team_adv = transition_batch.advantage
    team_adv = (team_adv - team_adv.mean()) / (team_adv.std() + 1e-8)
    doer_old_log_probs = transition_batch.doer_log_prob

    seer_old_log_probs = doer_old_log_probs.sum(axis=-1)
    seer_new_log_probs = doer_new_log_probs.sum(axis=-1)
    seer_logratio = seer_new_log_probs - seer_old_log_probs
    seer_ratio = jnp.exp(seer_logratio)
    seer_loss_unclipped = seer_ratio * team_adv
    seer_loss_clipped = jnp.clip(seer_ratio, 1.0 - clip_eps, 1.0 + clip_eps) * team_adv
    seer_actor_loss = -jnp.minimum(seer_loss_unclipped, seer_loss_clipped).mean()

    doer_logratio = doer_new_log_probs - doer_old_log_probs
    doer_ratio = jnp.exp(doer_logratio)
    team_adv_expanded = team_adv[..., None]
    doer_loss_unclipped = doer_ratio * team_adv_expanded
    doer_loss_clipped = (
        jnp.clip(doer_ratio, 1.0 - clip_eps, 1.0 + clip_eps) * team_adv_expanded
    )
    doer_actor_loss = -jnp.minimum(doer_loss_unclipped, doer_loss_clipped).mean()

    message_entropy, message_entropy_normalized, dominant_code_prob = (
        _compute_message_entropy_metrics(discrete_messages, message_levels)
    )
    seer_loss = seer_actor_loss - seer_entropy_coef * message_entropy
    doer_loss = doer_actor_loss - entropy_coef * doer_entropy

    return (seer_loss, doer_loss), {
        "seer_actor_loss": seer_actor_loss,
        "doer_actor_loss": doer_actor_loss,
        "entropy": doer_entropy,
        "seer_ratio": seer_ratio.mean(),
        "doer_ratio": doer_ratio.mean(),
        "message_entropy": message_entropy,
        "message_entropy_normalized": message_entropy_normalized,
        "message_dominant_probability": dominant_code_prob,
        "discrete_messages": discrete_messages,
    }


def calculate_two_doer_critic_loss(
    critic_apply_fn: Callable,
    critic_params: Any,
    transition_batch: TwoDoerTransition,
    value_clip: float = 0.2,
) -> Tuple[jnp.ndarray, dict]:
    values = critic_apply_fn({"params": critic_params}, transition_batch.global_obs).squeeze(-1)
    value_pred_clipped = transition_batch.value + jnp.clip(
        values - transition_batch.value,
        -value_clip,
        value_clip,
    )
    value_losses = jnp.square(values - transition_batch.return_val)
    value_losses_clipped = jnp.square(value_pred_clipped - transition_batch.return_val)
    critic_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
    return critic_loss, {"critic_loss": critic_loss}

# 3. The Update Step: JIT-compiled gradient application
import functools

@functools.partial(
    jax.jit,
    static_argnames=(
        "seer_apply_fn",
        "doer_apply_fn",
        "message_levels",
        "num_ppo_epochs",
        "num_minibatches",
    ),
)
def update_actor(
    actor_state: TrainState, 
    transition_batch: Transition,
    init_seer_carry: Any,
    init_doer_carry: Any,
    seer_apply_fn: Callable,
    doer_apply_fn: Callable,
    rng: jax.random.PRNGKey,
    control_mode: jnp.ndarray,
    message_levels: Tuple[int, ...],
    seer_entropy_coef: jnp.ndarray,
    num_ppo_epochs: int = 4,
    num_minibatches: int = 1
) -> Tuple[TrainState, dict]:
    """Computes gradients and updates the actor network using PPO epochs."""
    
    # Add a batch dimension if missing (assumes trajectory is num_steps, ...)
    # Wait, the prompt says trajectory is (batch_size, seq_len, ...)
    # If the user scales num_envs later, batch_size = num_envs
    
    batch_size = transition_batch.doer_action.shape[0]
    minibatch_size = batch_size // num_minibatches
    
    def epoch_fn(carry, _):
        actor_state, key = carry
        key, subkey = jax.random.split(key)
        
        # Shuffle along the batch dimension
        permutation = jax.random.permutation(subkey, batch_size)
        
        def minibatch_fn(state, start_idx):
            # Slice the minibatch
            indices = jax.lax.dynamic_slice_in_dim(permutation, start_idx, minibatch_size)
            mb_transition = jax.tree_util.tree_map(lambda x: x[indices], transition_batch)
            
            # Since calculate_actor_loss currently assumes scan over time, and time is dim 1:
            # wait, if input is (batch, time, ...) and scan is over time, we must swap axes!
            # scan_fn expects transition sequence to be the leading dimension.
            # So let's swap seq_len (dim 1) to be dim 0 for scan.
            mb_transition_time_first = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), mb_transition)
            
            # Extract initial carries for this minibatch
            # Assuming init_seer_carry is (batch_size, ...)
            mb_seer_carry = jax.tree_util.tree_map(lambda x: x[indices], init_seer_carry)
            mb_doer_carry = jax.tree_util.tree_map(lambda x: x[indices], init_doer_carry)
            
            def seer_loss_fn(seer_params):
                (seer_loss, _), metrics = calculate_actor_losses(
                    seer_apply_fn,
                    doer_apply_fn,
                    {"seer": seer_params, "doer": state.params["doer"]},
                    mb_transition_time_first,
                    mb_seer_carry,
                    mb_doer_carry,
                    control_mode=control_mode,
                    message_levels=message_levels,
                    seer_entropy_coef=seer_entropy_coef,
                )
                return seer_loss, metrics

            def doer_loss_fn(doer_params):
                (_, doer_loss), metrics = calculate_actor_losses(
                    seer_apply_fn,
                    doer_apply_fn,
                    {"seer": state.params["seer"], "doer": doer_params},
                    mb_transition_time_first,
                    mb_seer_carry,
                    mb_doer_carry,
                    control_mode=control_mode,
                    message_levels=message_levels,
                    seer_entropy_coef=seer_entropy_coef,
                )
                return doer_loss, metrics

            (seer_loss, seer_metrics), seer_grads = jax.value_and_grad(
                seer_loss_fn,
                has_aux=True,
            )(state.params["seer"])
            (doer_loss, doer_metrics), doer_grads = jax.value_and_grad(
                doer_loss_fn,
                has_aux=True,
            )(state.params["doer"])

            grads = {"seer": seer_grads, "doer": doer_grads}

            # Record explicit gradient norms for auditing
            seer_grad_norm = optax.global_norm(grads["seer"])
            doer_grad_norm = optax.global_norm(grads["doer"])

            metrics = {
                "seer_loss": seer_loss,
                "doer_loss": doer_loss,
                "seer_actor_loss": seer_metrics["seer_actor_loss"],
                "doer_actor_loss": doer_metrics["doer_actor_loss"],
                "entropy": doer_metrics["entropy"],
                "seer_ratio": seer_metrics["seer_ratio"],
                "doer_ratio": doer_metrics["doer_ratio"],
                "message_entropy": seer_metrics["message_entropy"],
                "message_entropy_normalized": seer_metrics["message_entropy_normalized"],
                "message_dominant_probability": seer_metrics["message_dominant_probability"],
                "seer_nav_entropy": seer_metrics["seer_nav_entropy"],
                "discrete_messages": seer_metrics["discrete_messages"],
                "seer_grad_norm": seer_grad_norm,
                "doer_grad_norm": doer_grad_norm,
                "actor_loss": seer_loss + doer_loss,
            }
            
            new_state = state.apply_gradients(grads=grads)
            return new_state, metrics
            
        # Minibatch loop (scan over start_indices)
        start_indices = jnp.arange(0, batch_size, minibatch_size)
        actor_state, mb_metrics = jax.lax.scan(minibatch_fn, actor_state, start_indices)
        
        # Average metrics over minibatches
        epoch_metrics = {k: v.mean() if k != "discrete_messages" else v[0] for k, v in mb_metrics.items()}
        return (actor_state, key), epoch_metrics
        
    (final_actor_state, _), epoch_metrics = jax.lax.scan(
        epoch_fn, (actor_state, rng), None, length=num_ppo_epochs
    )
    
    # Return averaged metrics over epochs
    final_metrics = {k: v.mean() if k != "discrete_messages" else v[0] for k, v in epoch_metrics.items()}
    return final_actor_state, final_metrics

@functools.partial(jax.jit, static_argnames=("critic_apply_fn", "num_ppo_epochs", "num_minibatches"))
def update_critic(
    critic_state: TrainState, 
    transition_batch: Transition,
    critic_apply_fn: Callable,
    rng: jax.random.PRNGKey,
    num_ppo_epochs: int = 4,
    num_minibatches: int = 1
) -> Tuple[TrainState, dict]:
    """Computes gradients and updates the critic network."""
    
    # For critic we can flatten the batch and sequence dimension since it's just MLPs
    batch_size = transition_batch.doer_action.shape[0] * transition_batch.doer_action.shape[1]
    flat_transition = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), transition_batch)
    minibatch_size = batch_size // num_minibatches
    
    def epoch_fn(carry, _):
        critic_state, key = carry
        key, subkey = jax.random.split(key)
        permutation = jax.random.permutation(subkey, batch_size)
        
        def minibatch_fn(state, start_idx):
            indices = jax.lax.dynamic_slice_in_dim(permutation, start_idx, minibatch_size)
            mb_transition = jax.tree_util.tree_map(lambda x: x[indices], flat_transition)
            
            loss_fn = lambda params: calculate_critic_loss(
                critic_apply_fn, params, mb_transition
            )
            
            (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            new_state = state.apply_gradients(grads=grads)
            return new_state, metrics
            
        start_indices = jnp.arange(0, batch_size, minibatch_size)
        critic_state, mb_metrics = jax.lax.scan(minibatch_fn, critic_state, start_indices)
        
        epoch_metrics = jax.tree_util.tree_map(lambda x: x.mean(), mb_metrics)
        return (critic_state, key), epoch_metrics

    (final_critic_state, _), epoch_metrics = jax.lax.scan(
        epoch_fn, (critic_state, rng), None, length=num_ppo_epochs
    )
    
    final_metrics = jax.tree_util.tree_map(lambda x: x.mean(), epoch_metrics)
    return final_critic_state, final_metrics


@functools.partial(
    jax.jit,
    static_argnames=(
        "seer_apply_fn",
        "doer_apply_fn",
        "message_levels",
        "num_ppo_epochs",
        "num_minibatches",
    ),
)
def update_actor_two_doer(
    actor_state: TrainState,
    transition_batch: TwoDoerTransition,
    init_seer_carry: Any,
    init_doer_carry: Any,
    seer_apply_fn: Callable,
    doer_apply_fn: Callable,
    rng: jax.random.PRNGKey,
    message_levels: Tuple[int, ...],
    seer_entropy_coef: jnp.ndarray,
    num_ppo_epochs: int = 4,
    num_minibatches: int = 1,
) -> Tuple[TrainState, dict]:
    batch_size = transition_batch.doer_action.shape[0]
    minibatch_size = batch_size // num_minibatches

    def epoch_fn(carry, _):
        actor_state, key = carry
        key, subkey = jax.random.split(key)
        permutation = jax.random.permutation(subkey, batch_size)

        def minibatch_fn(state, start_idx):
            indices = jax.lax.dynamic_slice_in_dim(permutation, start_idx, minibatch_size)
            mb_transition = jax.tree_util.tree_map(lambda x: x[indices], transition_batch)
            mb_transition_time_first = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(x, 0, 1),
                mb_transition,
            )
            mb_seer_carry = jax.tree_util.tree_map(lambda x: x[indices], init_seer_carry)
            mb_doer_carry = jax.tree_util.tree_map(lambda x: x[indices], init_doer_carry)

            def seer_loss_fn(seer_params):
                (seer_loss, _), metrics = calculate_two_doer_actor_losses(
                    seer_apply_fn,
                    doer_apply_fn,
                    {"seer": seer_params, "doer": state.params["doer"]},
                    mb_transition_time_first,
                    mb_seer_carry,
                    mb_doer_carry,
                    message_levels=message_levels,
                    seer_entropy_coef=seer_entropy_coef,
                )
                return seer_loss, metrics

            def doer_loss_fn(doer_params):
                (_, doer_loss), metrics = calculate_two_doer_actor_losses(
                    seer_apply_fn,
                    doer_apply_fn,
                    {"seer": state.params["seer"], "doer": doer_params},
                    mb_transition_time_first,
                    mb_seer_carry,
                    mb_doer_carry,
                    message_levels=message_levels,
                    seer_entropy_coef=seer_entropy_coef,
                )
                return doer_loss, metrics

            (seer_loss, seer_metrics), seer_grads = jax.value_and_grad(
                seer_loss_fn,
                has_aux=True,
            )(state.params["seer"])
            (doer_loss, doer_metrics), doer_grads = jax.value_and_grad(
                doer_loss_fn,
                has_aux=True,
            )(state.params["doer"])
            grads = {"seer": seer_grads, "doer": doer_grads}
            seer_grad_norm = optax.global_norm(grads["seer"])
            doer_grad_norm = optax.global_norm(grads["doer"])
            metrics = {
                "seer_loss": seer_loss,
                "doer_loss": doer_loss,
                "seer_actor_loss": seer_metrics["seer_actor_loss"],
                "doer_actor_loss": doer_metrics["doer_actor_loss"],
                "entropy": doer_metrics["entropy"],
                "seer_ratio": seer_metrics["seer_ratio"],
                "doer_ratio": doer_metrics["doer_ratio"],
                "message_entropy": seer_metrics["message_entropy"],
                "message_entropy_normalized": seer_metrics["message_entropy_normalized"],
                "message_dominant_probability": seer_metrics["message_dominant_probability"],
                "discrete_messages": seer_metrics["discrete_messages"],
                "seer_grad_norm": seer_grad_norm,
                "doer_grad_norm": doer_grad_norm,
                "actor_loss": seer_loss + doer_loss,
            }
            new_state = state.apply_gradients(grads=grads)
            return new_state, metrics

        start_indices = jnp.arange(0, batch_size, minibatch_size)
        actor_state, mb_metrics = jax.lax.scan(minibatch_fn, actor_state, start_indices)
        epoch_metrics = {k: v.mean() if k != "discrete_messages" else v[0] for k, v in mb_metrics.items()}
        return (actor_state, key), epoch_metrics

    (final_actor_state, _), epoch_metrics = jax.lax.scan(
        epoch_fn,
        (actor_state, rng),
        None,
        length=num_ppo_epochs,
    )
    final_metrics = {k: v.mean() if k != "discrete_messages" else v[0] for k, v in epoch_metrics.items()}
    return final_actor_state, final_metrics


@functools.partial(jax.jit, static_argnames=("critic_apply_fn", "num_ppo_epochs", "num_minibatches"))
def update_critic_two_doer(
    critic_state: TrainState,
    transition_batch: TwoDoerTransition,
    critic_apply_fn: Callable,
    rng: jax.random.PRNGKey,
    num_ppo_epochs: int = 4,
    num_minibatches: int = 1,
) -> Tuple[TrainState, dict]:
    batch_size = transition_batch.doer_action.shape[0] * transition_batch.doer_action.shape[1]
    flat_transition = jax.tree_util.tree_map(
        lambda x: x.reshape((batch_size,) + x.shape[2:]),
        transition_batch,
    )
    minibatch_size = batch_size // num_minibatches

    def epoch_fn(carry, _):
        critic_state, key = carry
        key, subkey = jax.random.split(key)
        permutation = jax.random.permutation(subkey, batch_size)

        def minibatch_fn(state, start_idx):
            indices = jax.lax.dynamic_slice_in_dim(permutation, start_idx, minibatch_size)
            mb_transition = jax.tree_util.tree_map(lambda x: x[indices], flat_transition)
            loss_fn = lambda params: calculate_two_doer_critic_loss(
                critic_apply_fn,
                params,
                mb_transition,
            )
            (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            new_state = state.apply_gradients(grads=grads)
            return new_state, metrics

        start_indices = jnp.arange(0, batch_size, minibatch_size)
        critic_state, mb_metrics = jax.lax.scan(minibatch_fn, critic_state, start_indices)
        epoch_metrics = jax.tree_util.tree_map(lambda x: x.mean(), mb_metrics)
        return (critic_state, key), epoch_metrics

    (final_critic_state, _), epoch_metrics = jax.lax.scan(
        epoch_fn,
        (critic_state, rng),
        None,
        length=num_ppo_epochs,
    )
    final_metrics = jax.tree_util.tree_map(lambda x: x.mean(), epoch_metrics)
    return final_critic_state, final_metrics
