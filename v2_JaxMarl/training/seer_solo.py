''' Solo training loop for the seer
The Seer is trained alone to solve the navigation task before being paired with the Doer.
Once trained, its LSTM weights are transferred to the full Seer+Doer system. '''

import jax
import jax
import jax.numpy as jnp
import distrax
import functools
from typing import Tuple
from flax.training.train_state import TrainState
import optax
from training.gae import compute_gae
from models.seer_solo import SeerSolo
import navix as nx
from envs.navix_wrapper import NavixGridWrapper
import wandb

def make_seer_solo_rollout_step(env, seer_apply_fn):
    """
    A closure that returns the JAX-compilable step function for solo Seer training.
    The Seer acts directly in the environment without a Doer or Critic.
    """

    def rollout_step(runner_state: Tuple, _):
        params, seer_carry, env_state, env_obs, rng = runner_state
        num_envs = env_obs["global_map"].shape[0]

        rng, seer_rng, env_rng = jax.random.split(rng, 3)
        env_step_keys = jax.random.split(env_rng, num_envs)

        # 1. Seer Forward Pass
        global_map = env_obs["global_map"]
        symbolic_obs = env_obs["symbolic_state"]
        next_seer_carry, action_logits = seer_apply_fn(
            {"params": params}, seer_carry, global_map, symbolic_obs
        )

        # 2. Action Selection
        pi = distrax.Categorical(logits=action_logits)
        action = pi.sample(seed=seer_rng)
        log_prob = pi.log_prob(action)

        # 3. Environment Step
        next_env_obs, next_env_state, reward, done, info = env.step_batch(
            env_step_keys, env_state, action
        )

        # 4. Reset carry on episode end
        done_mask = done[:, None]
        next_seer_carry = jax.tree_util.tree_map(
            lambda x: jnp.where(done_mask, jnp.zeros_like(x), x),
            next_seer_carry,
        )

        next_runner_state = (params, next_seer_carry, next_env_state, next_env_obs, rng)
        transition = (global_map, symbolic_obs, action, log_prob, reward, done)
        return next_runner_state, transition

    return rollout_step

@functools.partial(jax.jit, static_argnames=("num_steps", "step_fn"))
def generate_seer_solo_trajectory(
    params, rng, env_obs, env_state, seer_carry, num_steps: int, step_fn
):
    """
    Executes the full rollout and computes GAE for solo Seer training.
    """
    initial_runner_state = (params, seer_carry, env_state, env_obs, rng)

    # 1. Collect trajectory
    final_runner_state, trajectory = jax.lax.scan(
        step_fn, initial_runner_state, None, length=num_steps
    )

    global_map, symbolic_obs, action, log_prob, reward, done = trajectory

    # 2. Bootstrap value using last reward
    last_val = jnp.zeros_like(reward[-1])

    # 3. Compute GAE
    advantages, returns = jax.vmap(
        compute_gae, in_axes=(1, 1, 1, 0, None, None), out_axes=1
    )(reward, jnp.zeros_like(reward), done, last_val, 0.99, 0.95)

    return final_runner_state, (global_map, symbolic_obs, action, log_prob, reward, done, advantages, returns)

@functools.partial(jax.jit, static_argnames=("seer_apply_fn",))
def update_seer_solo(
    actor_state: TrainState,
    trajectory: Tuple,
    init_seer_carry: Tuple,
    seer_apply_fn,
    rng: jax.random.PRNGKey,
    clip_eps: float = 0.2,
    entropy_coef: float = 0.01,
) -> Tuple[TrainState, dict]:
    """
    PPO update for the solo Seer.
    """
    global_map, symbolic_obs, action, old_log_prob, reward, done, advantages, returns = trajectory

    def loss_fn(params):
        def scan_fn(carry, x):
            seer_carry = carry
            g_map, sym_obs = x
            next_seer_carry, action_logits = seer_apply_fn(
                {"params": params}, seer_carry, g_map, sym_obs
            )
            return next_seer_carry, action_logits

        _, action_logits = jax.lax.scan(
            scan_fn, init_seer_carry, (global_map, symbolic_obs)
        )

        pi = distrax.Categorical(logits=action_logits)
        new_log_prob = pi.log_prob(action)
        entropy = pi.entropy().mean()

        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        ratio = jnp.exp(new_log_prob - old_log_prob)

        loss_unclipped = ratio * adv
        loss_clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
        actor_loss = -jnp.minimum(loss_unclipped, loss_clipped).mean()

        total_loss = actor_loss - entropy_coef * entropy
        return total_loss, {"actor_loss": actor_loss, "entropy": entropy}

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(actor_state.params)
    new_actor_state = actor_state.apply_gradients(grads=grads)
    return new_actor_state, metrics
