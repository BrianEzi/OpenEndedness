import jax
import jax.numpy as jnp
import distrax
from typing import Tuple, Any, Dict
from training.gae import compute_gae 

# Assuming Transition is imported from your mappo.py or a shared datatypes file
from agents.mappo import Transition, TwoDoerTransition
from eval.metrics import compute_cic

def make_rollout_step(
    env,
    seer_apply_fn,
    doer_apply_fn,
    critic_apply_fn,
    follow_reward_scale=0.1,
):
    """
    A closure that returns the JAX-compilable step function.
    Passing the environment and apply functions here avoids passing them 
    repeatedly into the compiled loop.
    """
    follow_reward_scale = jnp.asarray(follow_reward_scale, dtype=jnp.float32)
    
    def rollout_step(runner_state: Tuple, _):
        """
        Executes a single environment step and network forward pass.
        Designed to be passed directly to jax.lax.scan.
        """
        # Unpack the runner state
        (
            params,
            seer_carry,
            doer_carry,
            env_state,
            env_obs,
            rng,
            vision_radius,
            control_mode,
            fixed_goal_position,
            fixed_start_position,
        ) = runner_state
        num_envs = env_obs["global_map"].shape[0]
        communication_mode = control_mode == 1
        
        # Split the PRNG key for the stochastic actions
        rng, seer_rng, doer_rng, env_rng = jax.random.split(rng, 4)
        env_step_keys = jax.random.split(env_rng, num_envs)
        
        # 1. Seer Forward Pass (Prefrontal Cortex)
        # Enforcing CTDE: Seer gets the global view[cite: 131, 132].
        # In a custom jaxmarl wrapper, you would extract these from env_obs
        global_map = env_obs["global_map"]
        symbolic_obs = env_obs["symbolic_state"]
        
        next_seer_carry, discrete_message, _, seer_nav_logits = seer_apply_fn(
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
        _, null_action_logits = doer_apply_fn(
            {"params": params["doer"]},
            doer_carry,
            local_obs,
            proprioception,
            jnp.zeros_like(discrete_message),
        )
        
        # 3. Action Selection
        seer_pi = distrax.Categorical(logits=seer_nav_logits)
        pi = distrax.Categorical(logits=action_logits)
        seer_action = seer_pi.sample(seed=seer_rng)
        doer_action = pi.sample(seed=doer_rng)
        seer_log_prob = seer_pi.log_prob(seer_action)
        doer_log_prob = pi.log_prob(doer_action)
        env_action = jnp.where(communication_mode, doer_action, seer_action)
        message_action = jnp.argmax(action_logits, axis=-1)
        null_message_action = jnp.argmax(null_action_logits, axis=-1)
        action_change_bonus = (
            message_action != null_message_action
        ).astype(jnp.float32) * follow_reward_scale
        
        # 4. Critic Forward Pass (Centralized Training)
        # The critic evaluates the global state to guide learning[cite: 111].
        value = critic_apply_fn({"params": params["critic"]}, global_map)
        
        # 5. Environment Step
        # Step the jaxmarl environment using the chosen action
        # Note: jaxmarl expects a dictionary of actions for multi-agent, 
        # adapt this based on your specific wrapper implementation.
        next_env_obs, next_env_state, reward, done, info = env.step_batch(
            env_step_keys,
            env_state,
            env_action,
            vision_radius=vision_radius,
            control_mode=control_mode,
            fixed_goal_position=fixed_goal_position,
            fixed_start_position=fixed_start_position,
        )

        task_reward = info["task_reward"]
        progress_reward = info["progress_reward"]
        step_penalty = info["step_penalty"]
        bump_penalty = info["bump_penalty"]
        useful_communication = jnp.logical_or(
            progress_reward > 0.0,
            task_reward > 0.0,
        )
        follow_reward = jnp.where(
            jnp.logical_and(communication_mode, useful_communication),
            action_change_bonus,
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        shared_comm_reward = (
            task_reward + progress_reward + follow_reward - step_penalty - bump_penalty
        )
        seer_reward = jnp.where(
            communication_mode,
            shared_comm_reward,
            task_reward + progress_reward - step_penalty - bump_penalty,
        )
        doer_reward = jnp.where(
            communication_mode,
            shared_comm_reward,
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        reward = jnp.stack([seer_reward, doer_reward], axis=-1)

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
            doer_action=doer_action,
            doer_log_prob=doer_log_prob,
            seer_action=seer_action,
            seer_log_prob=seer_log_prob,
            value=value,
            reward=reward,
            task_reward=task_reward,
            progress_reward=progress_reward,
            follow_reward=follow_reward,
            cic_reward_component=jnp.zeros_like(task_reward),
            cic_score=jnp.zeros_like(task_reward),
            step_penalty_component=step_penalty,
            bump_penalty_component=bump_penalty,
            done=done,
            # Advantage and return will be calculated post-rollout using GAE
            advantage=jnp.zeros_like(reward), 
            return_val=jnp.zeros_like(reward)
        )
        
        # Repack the updated runner state
        next_runner_state = (
            params,
            next_seer_carry,
            next_doer_carry,
            next_env_state,
            next_env_obs,
            rng,
            vision_radius,
            control_mode,
            fixed_goal_position,
            fixed_start_position,
        )
        
        return next_runner_state, transition

    return rollout_step

import functools

@functools.partial(jax.jit, static_argnames=("num_steps", "step_fn", "critic_apply_fn", "doer_apply_fn"))
def generate_trajectory_and_gae(
    params, rng, env_obs, env_state, seer_carry, doer_carry, vision_radius: jnp.ndarray, control_mode: jnp.ndarray, fixed_goal_position: jnp.ndarray, fixed_start_position: jnp.ndarray, cic_coef: jnp.ndarray, num_steps: int,
    step_fn, critic_apply_fn, doer_apply_fn
):
    """
    Executes the full episode rollout and computes GAE in a single compiled pass.
    Note: We pass the pre-compiled step_fn and initial states directly here 
    for better JAX compilation efficiency.
    """
    initial_runner_state = (
        params,
        seer_carry,
        doer_carry,
        env_state,
        env_obs,
        rng,
        vision_radius,
        control_mode,
        fixed_goal_position,
        fixed_start_position,
    )
    
    # 1. Execute the scan loop to collect the raw trajectory
    final_runner_state, trajectory_batch = jax.lax.scan(
        step_fn, initial_runner_state, None, length=num_steps
    )
    
    # 2. Extract the final state for Critic bootstrapping
    # Unpack the final runner state to get the last env_obs
    _, _, _, _, final_env_obs, final_rng, _, _, _, _ = final_runner_state
    
    # Enforce CTDE: The critic evaluates the global map 
    final_global_map = final_env_obs["global_map"]
    
    # 3. Calculate the bootstrap value (last_val)
    # The critic evaluates the state *after* the final step
    last_val = critic_apply_fn({"params": params["critic"]}, final_global_map)
    
    communication_mode = control_mode == 1

    def add_cic_bonus(_):
        cic_score = compute_cic(
            doer_apply_fn,
            params["doer"],
            trajectory_batch,
            doer_carry,
            final_rng,
        )
        per_step_cic_reward = cic_coef * cic_score / jnp.asarray(num_steps, dtype=jnp.float32)
        cic_reward_component = jnp.full_like(trajectory_batch.task_reward, per_step_cic_reward)
        cic_score_component = jnp.full_like(trajectory_batch.task_reward, cic_score)
        reward_with_cic = trajectory_batch.reward + cic_reward_component[..., None]
        return reward_with_cic, cic_reward_component, cic_score_component

    def skip_cic_bonus(_):
        zeros = jnp.zeros_like(trajectory_batch.task_reward)
        return trajectory_batch.reward, zeros, zeros

    reward_with_cic, cic_reward_component, cic_score_component = jax.lax.cond(
        jnp.logical_and(communication_mode, cic_coef > 0.0),
        add_cic_bonus,
        skip_cic_bonus,
        operand=None,
    )
    
    # 5. Compute GAE
    # Note: If you scale up to multiple environments (num_envs > 1), you would wrap 
    # compute_gae with jax.vmap(compute_gae, in_axes=1, out_axes=1) to vectorize 
    # the advantage calculation across all environments simultaneously.
    advantages, returns = jax.vmap(
        jax.vmap(
            compute_gae,
            in_axes=(1, 1, 1, 0, None, None),
            out_axes=1,
        ),
        in_axes=(2, 2, None, 1, None, None),
        out_axes=2,
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
        reward=reward_with_cic,
        cic_reward_component=cic_reward_component,
        cic_score=cic_score_component,
        advantage=advantages,
        return_val=returns
    )
    
    return final_runner_state, trajectory_batch


def make_two_doer_rollout_step(
    env,
    seer_apply_fn,
    doer_apply_fn,
    critic_apply_fn,
):
    """Rollout step for one Seer coordinating two embodied Doers."""

    def rollout_step(runner_state: Tuple, _):
        params, seer_carry, doer_carry, env_state, env_obs, rng, fixed_positions = runner_state
        num_envs = env_obs["global_map"].shape[0]
        global_map = env_obs["global_map"]
        symbolic_obs = env_obs["symbolic_state"]
        local_obs = env_obs["local_views"]
        proprioception = env_obs["proprioceptions"]

        rng, action_rng, env_rng = jax.random.split(rng, 3)
        env_step_keys = jax.random.split(env_rng, num_envs)

        next_seer_carry, discrete_messages, _, _ = seer_apply_fn(
            {"params": params["seer"]},
            seer_carry,
            global_map,
            symbolic_obs,
        )

        batch_size, num_doers = local_obs.shape[:2]
        flat_local_obs = local_obs.reshape((batch_size * num_doers,) + local_obs.shape[2:])
        flat_proprioception = proprioception.reshape(
            (batch_size * num_doers,) + proprioception.shape[2:]
        )
        flat_messages = discrete_messages.reshape(
            (batch_size * num_doers,) + discrete_messages.shape[2:]
        )
        flat_doer_carry = jax.tree_util.tree_map(
            lambda x: x.reshape((batch_size * num_doers,) + x.shape[2:]),
            doer_carry,
        )
        next_flat_doer_carry, flat_logits = doer_apply_fn(
            {"params": params["doer"]},
            flat_doer_carry,
            flat_local_obs,
            flat_proprioception,
            flat_messages,
        )
        next_doer_carry = jax.tree_util.tree_map(
            lambda x: x.reshape((batch_size, num_doers) + x.shape[1:]),
            next_flat_doer_carry,
        )
        doer_logits = flat_logits.reshape((batch_size, num_doers, flat_logits.shape[-1]))
        doer_pi = distrax.Categorical(logits=doer_logits)
        doer_action = doer_pi.sample(seed=action_rng)
        doer_log_prob = doer_pi.log_prob(doer_action)
        value = critic_apply_fn({"params": params["critic"]}, global_map).squeeze(-1)

        next_env_obs, next_env_state, reward, done, info = env.step_batch(
            env_step_keys,
            env_state,
            doer_action,
            fixed_positions=fixed_positions,
        )

        next_seer_carry = jax.tree_util.tree_map(
            lambda x: jnp.where(done[:, None], jnp.zeros_like(x), x),
            next_seer_carry,
        )
        next_doer_carry = jax.tree_util.tree_map(
            lambda x: jnp.where(done[:, None, None], jnp.zeros_like(x), x),
            next_doer_carry,
        )

        transition = TwoDoerTransition(
            global_obs=global_map,
            symbolic_obs=symbolic_obs,
            local_obs=local_obs,
            proprioception=proprioception,
            message=discrete_messages,
            doer_action=doer_action,
            doer_log_prob=doer_log_prob,
            value=value,
            reward=reward,
            task_reward=info["task_reward"],
            progress_reward_per_doer=info["progress_reward_per_doer"],
            step_penalty_component=info["step_penalty"],
            wall_penalty_component=info["wall_penalty"],
            collision_penalty_component=info["collision_penalty"],
            done=done,
            advantage=jnp.zeros_like(reward),
            return_val=jnp.zeros_like(reward),
        )

        next_runner_state = (
            params,
            next_seer_carry,
            next_doer_carry,
            next_env_state,
            next_env_obs,
            rng,
            fixed_positions,
        )
        return next_runner_state, transition

    return rollout_step


@functools.partial(
    jax.jit,
    static_argnames=("num_steps", "step_fn", "critic_apply_fn"),
)
def generate_two_doer_trajectory_and_gae(
    params,
    rng,
    env_obs,
    env_state,
    seer_carry,
    doer_carry,
    fixed_positions,
    num_steps: int,
    step_fn,
    critic_apply_fn,
):
    initial_runner_state = (
        params,
        seer_carry,
        doer_carry,
        env_state,
        env_obs,
        rng,
        fixed_positions,
    )
    final_runner_state, trajectory_batch = jax.lax.scan(
        step_fn,
        initial_runner_state,
        None,
        length=num_steps,
    )
    _, _, _, _, final_env_obs, _, _ = final_runner_state
    last_val = critic_apply_fn(
        {"params": params["critic"]},
        final_env_obs["global_map"],
    ).squeeze(-1)
    advantages, returns = jax.vmap(
        compute_gae,
        in_axes=(1, 1, 1, 0, None, None),
        out_axes=1,
    )(
        trajectory_batch.reward,
        trajectory_batch.value,
        trajectory_batch.done,
        last_val,
        0.99,
        0.95,
    )
    trajectory_batch = trajectory_batch.replace(
        advantage=advantages,
        return_val=returns,
    )
    return final_runner_state, trajectory_batch
