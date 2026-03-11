import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from typing import Callable, Dict, Any, Tuple

def compute_topographic_similarity(
    states: np.ndarray, 
    messages: np.ndarray, 
    metric: str = 'cosine'
) -> float:
    """
    Computes the Topographic Similarity (TopSim) between states and messages.
    A high score indicates a compositional language.
    
    Args:
        states: An array of flattened global states or semantic representations.
                Shape: (num_samples, state_dim).
        messages: An array of the generated FSQ message vectors.
                  Shape: (num_samples, message_dim).
        metric: The distance metric to use ('cosine', 'euclidean', etc.).
        
    Returns:
        rho: The Spearman rank correlation coefficient between the distance matrices.
    """
    # 1. Compute pairwise distance matrices
    # pdist returns a condensed 1D array of distances, which is perfect for correlation
    d_state = pdist(states, metric=metric)
    d_message = pdist(messages, metric=metric)
    
    # 2. Compute correlation
    # We use Spearman rank correlation as the relationship between state-space
    # distances and message-space distances is rarely perfectly linear.
    rho, _ = spearmanr(d_state, d_message)
    
    return float(rho)

def evaluate_causal_influence_ablation(
    env_step_fn: Callable,
    runner_state: Tuple,
    num_steps: int,
    noise_std: float = 5.0
) -> Dict[str, float]:
    """
    Executes an ablation study to measure the Causal Influence of Communication (CIC).
    Replaces the communication channel with Gaussian noise to test for causal dependency.
    
    Args:
        env_step_fn: The compiled JAX rollout step function.
        runner_state: The initialized state tuple for the rollout.
        num_steps: How many steps to run the evaluation.
        noise_std: The standard deviation of the Gaussian noise to inject.
        
    Returns:
        A dictionary comparing baseline reward to ablated reward.
    """
    # This is a conceptual template. In a full implementation, you would 
    # modify your env_step_fn to accept an 'ablation_mode' boolean flag.
    # When ablation_mode is True, the 'discrete_message' passed to the Doer
    # is replaced with jax.random.normal(rng, shape) * noise_std.
    
    # 1. Run baseline trajectory
    # final_state, baseline_traj = jax.lax.scan(env_step_fn, runner_state, None, num_steps)
    # baseline_reward = baseline_traj.reward.sum()
    
    # 2. Run ablated trajectory (with noise injected into the message channel)
    # final_state, ablated_traj = jax.lax.scan(ablated_step_fn, runner_state, None, num_steps)
    # ablated_reward = ablated_traj.reward.sum()
    
    # Placeholder return
    return {
        "baseline_return": 0.0,
        "ablated_return": 0.0,
        "causal_drop": 0.0 # baseline_return - ablated_return
    }

def check_zero_shot_generalization(
    executed_action: int, 
    target_object_id: int, 
    holdout_pairs: set
) -> bool:
    """
    Validates if the agent successfully executed a command on a held-out combination.
    
    Args:
        executed_action: The discrete action taken by the Doer.
        target_object_id: The ID of the object interacted with.
        holdout_pairs: A set of tuples defining the (action, object) pairs 
                       never seen during training.
                       
    Returns:
        True if the combination is in the holdout set, False otherwise.
    """
    combination = (executed_action, target_object_id)
    return combination in holdout_pairs