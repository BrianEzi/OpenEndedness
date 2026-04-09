import jax
import jax.numpy as jnp
from typing import Tuple

def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    last_val: jnp.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes Generalized Advantage Estimation (GAE) and target returns.
    
    Args:
        rewards: Array of rewards collected during the rollout. Shape: (num_steps,)
        values: Array of value predictions from the Critic. Shape: (num_steps,)
        dones: Array of boolean/integer done flags. Shape: (num_steps,)
        last_val: The Critic's value prediction for the state *after* the final step.
        gamma: Discount factor.
        gae_lambda: Bias-variance tradeoff parameter for GAE.
        
    Returns:
        advantages: The calculated GAE advantages. Shape: (num_steps,)
        returns: The target values for the Critic to learn from (Advantages + Values).
    """
    
    def _gae_step(carry, transition_data):
        """A single step of the reverse scan."""
        gae_t_plus_1 = carry
        reward, value, done, next_value = transition_data
        
        # Calculate the Temporal Difference (TD) error
        # If the episode ended (done=1), the next state has no value.
        delta = reward + gamma * next_value * (1.0 - done) - value
        
        # Calculate the advantage for the current timestep
        gae_t = delta + gamma * gae_lambda * (1.0 - done) * gae_t_plus_1
        
        # Pass the current advantage back as the carry for the previous timestep
        return gae_t, gae_t

    # To calculate the TD error, we need the value of the next state for every step.
    # We create an array of "next values" by shifting the values array by one 
    # and appending the bootstrap 'last_val' at the end.
    next_values = jnp.append(values[1:], last_val)
    
    # Pack the data for the scan
    scan_data = (rewards, values, dones, next_values)
    
    # Initialize the carry with 0.0 (the advantage after the final step)
    initial_gae = 0.0
    
    # Run the scan in reverse to propagate advantages backwards
    _, advantages = jax.lax.scan(_gae_step, initial_gae, scan_data, reverse=True)
    
    # The return value (target for the critic) is simply the advantage + the predicted value
    returns = advantages + values
    
    return advantages, returns