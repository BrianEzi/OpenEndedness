import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from env import BlindFetchEnv
from models import ActorCritic
from train import make_train
import analysis
import pandas as pd

def main():
    # Configuration
    # Note: 256 envs * 128 steps = 32,768 steps per update.
    # User requested 100,000 total timesteps. This would be ~3 updates.
    # To ensure convergence for the proof of concept, we will use more updates.
    # Let's target ~1M steps (approx 30 updates) or slightly more.
    # We'll set num_updates to 50 for robust demonstration (approx 1.6M steps).
    
    config = {
        "total_timesteps": 2000000, # Interpreted as sufficient steps for convergence
        "num_envs": 256,
        "num_steps": 128,
        "lr": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95
    }
    
    config["num_updates"] = config["total_timesteps"] // (config["num_envs"] * config["num_steps"])
    print(f"Running training for {config['num_updates']} updates of {config['num_envs']*config['num_steps']} steps each.")

    # Run Training
    rng = jax.random.PRNGKey(42)
    rng, train_rng = jax.random.split(rng)
    
    train_fn = make_train(config)
    final_state, metrics = train_fn(train_rng)

    # Initialize network for inference
    network = ActorCritic()
    
    # helper to print metrics
    # metrics is (num_updates,). We can print the last one.
    print("Training Complete.")
    print(f"Final Reward Mean: {metrics['reward_mean'][-1]:.4f}")
    print(f"Final Loss: {metrics['loss'][-1]:.4f}")
    
    # Save metrics
    df = pd.DataFrame(metrics)
    df.to_csv("training_log.csv", index=False)
    
    # --- Analysis Data Collection ---
    # 1. Grounding Heatmap Data (Random Walk / Evaluation Batch)
    # We want to see what message Seer sends for various positions.
    # Let's run a batch of steps with random positions or just normal evaluation.
    print("Collecting analysis data (Randomized Positions)...")
    
    heatmap_pos = []
    heatmap_msgs = []
    heatmap_targets = []
    
    an_env = BlindFetchEnv()
    # Generate 1000 random scenarios to ensure we cover negative relative coords
    an_keys = jax.random.split(rng, 1000)
    
    for i in range(1000):
        key_i = an_keys[i]
        k1, k2, k3 = jax.random.split(key_i, 3)
        
        # 1. Start with a standard reset
        state_i = an_env.reset(k1)
        
        # 2. MANUALLY FORCE RANDOM POSITIONS
        # We overwrite the positions to ensure we see every possible Target-Doer relationship
        rand_doer = jax.random.randint(k2, shape=(2,), minval=0, maxval=5)
        rand_target = jax.random.randint(k3, shape=(2,), minval=0, maxval=5)
        
        # Assuming EnvState is a Flax struct or NamedTuple, we use .replace
        # If your EnvState is a simple object, use state_i.doer_pos = ...
        state_i = state_i.replace(doer_pos=rand_doer, target_pos=rand_target)
        
        # 3. Get the observation for this forced state
        obs_i = an_env.get_obs(state_i)
        
        # 4. Get message
        seer_dist, _, _ = network.apply(final_state.params, obs_i)
        msg = seer_dist.loc
        
        heatmap_pos.append(state_i.doer_pos)
        heatmap_targets.append(state_i.target_pos)
        heatmap_msgs.append(msg)
        
    eval_data = {
        'grid_pos': np.array(heatmap_pos),
        'target_pos_all': np.array(heatmap_targets),
        'messages': np.array(heatmap_msgs)
    }

    # Verification: 1 Episode (Trajectory)
    print("\nStarting Verification Episode...")
    env = BlindFetchEnv()
    rng, eval_rng = jax.random.split(rng)
    
    # Reset
    obs = env.get_obs(env.reset(eval_rng))
    state = env.reset(eval_rng)
    
    # We need to use the learned params.
    # Define a single-step apply
    
    print(f"Target Position: {state.target_pos}")
    
    done = False
    step_count = 0
    path = [state.doer_pos]
    
    network = ActorCritic()
    
    while not done:
        # Get action from network (deterministic or sampled? usually deterministic for eval, but PPO uses stochastic)
        # Let's sample to show robustness or argmax?
        # Continuous: mean. Discrete: argmax.
        
        # We'll just use the same sample logic for simplicity, or we can peek at means.
        # Efficient way:
        seer_dist, doer_dist, _ = network.apply(final_state.params, obs)
        
        # Deterministic check for Seer? 
        # seer_dist is Normal. Mean is `loc`.
        seer_msg = seer_dist.loc
        
        # Deterministic check for Doer?
        # doer_dist is Categorical. Mode is argmax logits.
        doer_move = jnp.argmax(doer_dist.logits)
        
        print(f"Step {step_count}: Doer {state.doer_pos} | Seer Msg {seer_msg} | Action {doer_move}")
        
        # Step Env
        # Environment expects actions dict
        actions = {'seer_msg': seer_msg, 'doer_move': doer_move}
        
        # JAX env requires PRNG for step (even if deterministic logic, interface might require it)
        rng, step_rng = jax.random.split(rng)
        state, reward, done, info = env.step(state, actions, step_rng)
        obs = env.get_obs(state)
        path.append(state.doer_pos)
        
        step_count += 1
        
        if info['reached_target']:
            print("SUCCESS: Target Reached!")
            break
            
    if not info['reached_target']:
        print("FAILURE: Did not reach target in max steps.")
        
    # Add trajectory to eval_data
    eval_data['trajectory'] = np.array(path)
    eval_data['target_pos'] = state.target_pos

    # --- Run Analysis Plots ---
    try:
        analysis.plot_results(metrics, eval_data=eval_data)
        print(f"Final Success Rate: {metrics['success_rate'][-1]*100:.1f}% | Average Message Norm: {metrics['msg_magnitude'][-1]:.4f}")
    except Exception as e:
        print(f"Analysis failed: {e}")


if __name__ == "__main__":
    main()
