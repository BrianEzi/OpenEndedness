import jax
import jax.numpy as jnp
import distrax
from typing import Tuple, Any, Dict
from training.gae import compute_gae 

# Assuming Transition is imported from your mappo.py or a shared datatypes file
from agents.mappo import Transition 
from eval.metrics import compute_cic

def make_rollout_step(env, seer_apply_fn, doer_apply_fn, critic_apply_fn):
    """
    A closure that returns the JAX-compilable step function.
    Passing the environment and apply functions here avoids passing them 
    repeatedly into the compiled loop.
    """
    
    def rollout_step(runner_state: Tuple, _):
        """
        Executes a single environment step and network forward pass.
        Designed to be passed directly to jax.lax.scan.
        """
        # Unpack the runner state
        params, seer_carry, doer_carry, env_state, env_obs, rng, vision_radius = runner_state
        num_envs = env_obs["global_map"].shape[0]
        
        # Split the PRNG key for the stochastic actions
        rng, doer_rng, env_rng = jax.random.split(rng, 3)
        env_step_keys = jax.random.split(env_rng, num_envs)
        
        # 1. Seer Forward Pass (Prefrontal Cortex)
        # Enforcing CTDE: Seer gets the global view[cite: 131, 132].
        # In a custom jaxmarl wrapper, you would extract these from env_obs
        global_map = env_obs["global_map"]
        symbolic_obs = env_obs["symbolic_state"]
        
        next_seer_carry, discrete_message, _ = seer_apply_fn(
            {"params": params["seer"]}, 
            seer_carry, 
            global_map, 
            symbolic_obs
        )
        
        # 2. Doer Forward Pass (Motor Cortex)
        # Enforcing Functional Asymmetry: Doer gets local view and the message[cite: 137, 138].
        local_obs = env_obs["local_view"]
        proprioception = env_obs["proprioception"]
        
        next_doer_carry, action_logits = doer_apply_fn(
            {"params": params["doer"]}, 
            doer_carry, 
            local_obs, 
            proprioception, 
            discrete_message
        )
        
        # 3. Action Selection via Distrax
        pi = distrax.Categorical(logits=action_logits)
        action = pi.sample(seed=doer_rng)
        log_prob = pi.log_prob(action)
        
        # 4. Critic Forward Pass (Centralized Training)
        # The critic evaluates the global state to guide learning[cite: 111].
        value = critic_apply_fn({"params": params["critic"]}, global_map).squeeze(-1)
        
        # 5. Environment Step
        # Step the jaxmarl environment using the chosen action
        # Note: jaxmarl expects a dictionary of actions for multi-agent, 
        # adapt this based on your specific wrapper implementation.
        next_env_obs, next_env_state, reward, done, info = env.step_batch(
            env_step_keys, env_state, action, vision_radius=vision_radius
        )

        done_mask = done[:, None]
        next_seer_carry = jax.tree_util.tree_map(
            lambda x: jnp.where(done_mask, jnp.zeros_like(x), x),
            next_seer_carry,
        )
        next_doer_carry = jax.tree_util.tree_map(
            lambda x: jnp.where(done_mask, jnp.zeros_like(x), x),
            next_doer_carry,
        )
        
        # 6. Build the Transition
        transition = Transition(
            global_obs=global_map,
            symbolic_obs=symbolic_obs,
            local_obs=local_obs,
            proprioception=proprioception,
            message=discrete_message,
            action=action,
            log_prob=log_prob,
            value=value,
            reward=reward,
            done=done,
            # Advantage and return will be calculated post-rollout using GAE
            advantage=jnp.zeros_like(reward), 
            return_val=jnp.zeros_like(reward)
        )
        
        # Repack the updated runner state
        next_runner_state = (
            params, next_seer_carry, next_doer_carry, next_env_state, next_env_obs, rng, vision_radius
        )
        
        return next_runner_state, transition

    return rollout_step

import functools

@functools.partial(jax.jit, static_argnames=("num_steps", "step_fn", "critic_apply_fn", "doer_apply_fn"))
def generate_trajectory_and_gae(
    params, rng, env_obs, env_state, seer_carry, doer_carry, vision_radius: jnp.ndarray, num_steps: int,
    step_fn, critic_apply_fn, doer_apply_fn
):
    """
    Executes the full episode rollout and computes GAE in a single compiled pass.
    Note: We pass the pre-compiled step_fn and initial states directly here 
    for better JAX compilation efficiency.
    """
    initial_runner_state = (params, seer_carry, doer_carry, env_state, env_obs, rng, vision_radius)
    
    # 1. Execute the scan loop to collect the raw trajectory
    final_runner_state, trajectory_batch = jax.lax.scan(
        step_fn, initial_runner_state, None, length=num_steps
    )
    
    # 2. Extract the final state for Critic bootstrapping
    # Unpack the final runner state to get the last env_obs
    _, _, _, _, final_env_obs, _, _ = final_runner_state
    
    # Enforce CTDE: The critic evaluates the global map 
    final_global_map = final_env_obs["global_map"]
    
    # 3. Calculate the bootstrap value (last_val)
    # The critic evaluates the state *after* the final step
    last_val = critic_apply_fn({"params": params["critic"]}, final_global_map).squeeze(-1)
    
    # 4. Compute CIC and Intrinsic Reward
    rng, cic_rng = jax.random.split(rng)
    cic_score = compute_cic(
        doer_apply_fn,
        params["doer"],
        trajectory_batch,
        doer_carry,
        cic_rng
    )
    
    w_cic = 0.1
    reward_with_cic = trajectory_batch.reward + w_cic * cic_score
    
    # 5. Compute GAE
    # Note: If you scale up to multiple environments (num_envs > 1), you would wrap 
    # compute_gae with jax.vmap(compute_gae, in_axes=1, out_axes=1) to vectorize 
    # the advantage calculation across all environments simultaneously.
    advantages, returns = jax.vmap(
        compute_gae,
        in_axes=(1, 1, 1, 0, None, None),
        out_axes=1,
    )(
        reward_with_cic,
        trajectory_batch.value,
        trajectory_batch.done,
        last_val,
        0.99,
        0.95,
    )
    
    # 5. Update the trajectory batch
    # Using the .replace() method provided by chex/flax dataclasses
    trajectory_batch = trajectory_batch.replace(
        advantage=advantages,
        return_val=returns
    )
    
    return final_runner_state, trajectory_batch
