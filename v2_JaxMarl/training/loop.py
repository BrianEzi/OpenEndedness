import jax
import jax.numpy as jnp
import distrax
from typing import Tuple, Any, Dict

# Assuming Transition is imported from your mappo.py or a shared datatypes file
from agents.mappo import Transition 

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
        params, seer_carry, doer_carry, env_state, env_obs, rng = runner_state
        
        # Split the PRNG key for the stochastic actions
        rng, seer_rng, doer_rng = jax.random.split(rng, 3)
        
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
        value = critic_apply_fn({"params": params["critic"]}, global_map)
        
        # 5. Environment Step
        # Step the jaxmarl environment using the chosen action
        # Note: jaxmarl expects a dictionary of actions for multi-agent, 
        # adapt this based on your specific wrapper implementation.
        next_env_obs, next_env_state, reward, done, info = env.step(
            rng, env_state, action
        )
        
        # 6. Build the Transition
        transition = Transition(
            global_obs=global_map,
            local_obs=local_obs,
            action=action,
            log_prob=log_prob,
            value=value.squeeze(),
            reward=reward,
            done=done,
            # Advantage and return will be calculated post-rollout using GAE
            advantage=jnp.zeros_like(reward), 
            return_val=jnp.zeros_like(reward)
        )
        
        # Repack the updated runner state
        next_runner_state = (
            params, next_seer_carry, next_doer_carry, next_env_state, next_env_obs, rng
        )
        
        return next_runner_state, transition

    return rollout_step

def generate_trajectory(
    env, params, rng, num_steps: int, seer_apply_fn, doer_apply_fn, critic_apply_fn
):
    """Compiles and executes the full episode rollout."""
    
    # Initialize environment
    rng, env_rng = jax.random.split(rng)
    env_obs, env_state = env.reset(env_rng)
    
    # Initialize LSTM states (assuming a batch size of 1 for a single env instance)
    seer_carry = seer_apply_fn.initialize_carry(batch_size=1, hidden_size=128)
    doer_carry = doer_apply_fn.initialize_carry(batch_size=1, hidden_size=128)
    
    initial_runner_state = (params, seer_carry, doer_carry, env_state, env_obs, rng)
    
    # Create the compiled step function
    step_fn = make_rollout_step(env, seer_apply_fn, doer_apply_fn, critic_apply_fn)
    
    # Execute the scan loop
    # jax.lax.scan returns the final state and a stacked array of all transitions
    final_runner_state, trajectory_batch = jax.lax.scan(
        step_fn, initial_runner_state, None, length=num_steps
    )
    
    return final_runner_state, trajectory_batch